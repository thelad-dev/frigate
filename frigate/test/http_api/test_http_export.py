import io
import json
import os
import subprocess as sp
import tempfile
import zipfile
from unittest.mock import patch

from frigate.const import CLIPS_DIR
from frigate.jobs.export import (
    ExportJob,
    get_export_job_manager,
    reap_stale_exports,
    start_export_job,
)
from frigate.models import Export, ExportCase, Previews, Recordings
from frigate.test.http_api.base_http_test import AuthTestClient, BaseTestHttp


class TestHttpExport(BaseTestHttp):
    def setUp(self):
        super().setUp([Export, ExportCase, Previews, Recordings])
        self.minimal_config["cameras"]["backyard"] = {
            "ffmpeg": {
                "inputs": [{"path": "rtsp://10.0.0.2:554/video", "roles": ["detect"]}]
            },
            "detect": {
                "height": 1080,
                "width": 1920,
                "fps": 5,
            },
        }
        self.app = super().create_app()

    def tearDown(self):
        self.app.dependency_overrides.clear()
        super().tearDown()

    def _insert_recording(
        self,
        recording_id: str,
        camera: str,
        start_time: float,
        end_time: float,
    ) -> None:
        Recordings.create(
            id=recording_id,
            camera=camera,
            path=f"/tmp/{recording_id}.mp4",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            motion=0,
            objects=0,
            dBFS=0,
            segment_size=1,
            regions=0,
            motion_heatmap=[],
        )

    def test_create_export_case_uses_wall_clock_time(self):
        with patch("frigate.api.export.time.time", return_value=1234.5):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/cases",
                    json={
                        "name": "Investigation",
                        "description": "A test case",
                    },
                )

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["created_at"] == 1234.5
        assert response_json["updated_at"] == 1234.5

        case = ExportCase.get(ExportCase.id == response_json["id"])
        assert case.created_at.timestamp() == 1234.5
        assert case.updated_at.timestamp() == 1234.5

    def test_update_export_case_refreshes_updated_at(self):
        case = ExportCase.create(
            id="case123",
            name="Old name",
            description="Old description",
            created_at=10,
            updated_at=10,
        )

        with patch("frigate.api.export.time.time", return_value=2222.0):
            with AuthTestClient(self.app) as client:
                response = client.patch(
                    f"/cases/{case.id}",
                    json={"name": "New name", "description": "Updated"},
                )

        assert response.status_code == 200

        refreshed = ExportCase.get(ExportCase.id == case.id)
        assert refreshed.name == "New name"
        assert refreshed.description == "Updated"
        assert refreshed.updated_at.timestamp() == 2222.0

    def test_delete_export_case_delete_exports_cancels_queued_jobs(self):
        case = ExportCase.create(
            id="case_delete_me",
            name="Delete me",
            description="",
            created_at=10,
            updated_at=10,
        )
        other_case = ExportCase.create(
            id="case_keep_me",
            name="Keep me",
            description="",
            created_at=20,
            updated_at=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "case_export.mp4")
            thumb_path = os.path.join(tmpdir, "case_export.webp")
            other_video_path = os.path.join(tmpdir, "other_export.mp4")
            other_thumb_path = os.path.join(tmpdir, "other_export.webp")

            with open(video_path, "wb") as handle:
                handle.write(b"case")
            with open(thumb_path, "wb") as handle:
                handle.write(b"thumb")
            with open(other_video_path, "wb") as handle:
                handle.write(b"other")
            with open(other_thumb_path, "wb") as handle:
                handle.write(b"thumb")

            Export.create(
                id="export_in_case",
                camera="front_door",
                name="Case export",
                date=100,
                video_path=video_path,
                thumb_path=thumb_path,
                in_progress=False,
                export_case=case,
            )
            Export.create(
                id="export_other_case",
                camera="front_door",
                name="Other export",
                date=110,
                video_path=other_video_path,
                thumb_path=other_thumb_path,
                in_progress=False,
                export_case=other_case,
            )

            with (
                patch("frigate.jobs.export._job_manager", None),
                patch(
                    "frigate.jobs.export.ExportJobManager.ensure_started",
                    autospec=True,
                    return_value=None,
                ),
            ):
                start_export_job(
                    self.app.frigate_config,
                    ExportJob(
                        id="queued_case_job",
                        camera="front_door",
                        export_case_id=case.id,
                        request_start_time=100,
                        request_end_time=120,
                    ),
                )
                start_export_job(
                    self.app.frigate_config,
                    ExportJob(
                        id="queued_other_job",
                        camera="front_door",
                        export_case_id=other_case.id,
                        request_start_time=130,
                        request_end_time=150,
                    ),
                )

                manager = get_export_job_manager(self.app.frigate_config)
                assert {job.id for job in manager.list_active_jobs()} == {
                    "queued_case_job",
                    "queued_other_job",
                }

                with AuthTestClient(self.app) as client:
                    response = client.delete(f"/cases/{case.id}?delete_exports=true")

            assert response.status_code == 200
            assert ExportCase.get_or_none(ExportCase.id == case.id) is None
            assert ExportCase.get_or_none(ExportCase.id == other_case.id) is not None
            assert Export.get_or_none(Export.id == "export_in_case") is None
            assert Export.get_or_none(Export.id == "export_other_case") is not None
            assert not os.path.exists(video_path)
            assert not os.path.exists(thumb_path)

            cancelled_job = manager.get_job("queued_case_job")
            assert cancelled_job is not None
            assert cancelled_job.status == "cancelled"

            remaining_job = manager.get_job("queued_other_job")
            assert remaining_job is not None
            assert remaining_job.status == "queued"
            assert [job.id for job in manager.list_active_jobs()] == [
                "queued_other_job"
            ]

    def test_batch_export_creates_case_and_reports_partial_success(self):
        self._insert_recording("rec-front", "front_door", 100, 200)

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                                "friendly_name": "Incident - Front Door",
                            },
                            {
                                "camera": "backyard",
                                "start_time": 110,
                                "end_time": 150,
                                "friendly_name": "Incident - Backyard",
                            },
                        ],
                        "new_case_name": "Case Alpha",
                        "new_case_description": "Batch export",
                    },
                )

        assert response.status_code == 202
        response_json = response.json()
        assert len(response_json["export_ids"]) == 1
        assert response_json["results"] == [
            {
                "camera": "front_door",
                "export_id": response_json["export_ids"][0],
                "success": True,
                "status": "queued",
                "error": None,
                "item_index": 0,
                "client_item_id": None,
            },
            {
                "camera": "backyard",
                "export_id": None,
                "success": False,
                "status": None,
                "error": "No recordings found for time range",
                "item_index": 1,
                "client_item_id": None,
            },
        ]
        start_export_job.assert_called_once()

        case = ExportCase.get(ExportCase.id == response_json["export_case_id"])
        assert case.name == "Case Alpha"
        assert case.description == "Batch export"

    def test_single_export_is_queued_immediately(self):
        self._insert_recording("rec-front", "front_door", 100, 200)

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/export/front_door/start/110/end/150",
                    json={
                        "name": "Queued export",
                    },
                )

        assert response.status_code == 202
        response_json = response.json()
        assert response_json["success"] is True
        assert response_json["status"] == "queued"
        assert response_json["export_id"].startswith("front_door_")
        start_export_job.assert_called_once()

    def test_single_export_returns_503_when_queue_full(self):
        self._insert_recording("rec-front", "front_door", 100, 200)

        from frigate.jobs.export import ExportQueueFullError

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=ExportQueueFullError("Export queue is full"),
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/export/front_door/start/110/end/150",
                    json={
                        "name": "Rejected export",
                    },
                )

        assert response.status_code == 503
        response_json = response.json()
        assert response_json["success"] is False
        assert "queue is full" in response_json["message"].lower()

    def test_batch_export_returns_503_when_queue_cannot_fit_batch(self):
        self._insert_recording("rec-front", "front_door", 100, 200)
        self._insert_recording("rec-back", "backyard", 100, 200)

        with patch(
            "frigate.api.export.available_export_queue_slots",
            return_value=1,
        ):
            with patch(
                "frigate.api.export.start_export_job",
                side_effect=lambda _config, job: job.id,
            ) as start_export_job:
                with AuthTestClient(self.app) as client:
                    response = client.post(
                        "/exports/batch",
                        json={
                            "items": [
                                {
                                    "camera": "front_door",
                                    "start_time": 110,
                                    "end_time": 150,
                                },
                                {
                                    "camera": "backyard",
                                    "start_time": 110,
                                    "end_time": 150,
                                },
                            ],
                            "new_case_name": "Overflow Case",
                        },
                    )

        assert response.status_code == 503
        assert response.json()["success"] is False
        start_export_job.assert_not_called()

        # Empty case should NOT have been created
        assert ExportCase.select().count() == 0

    def test_get_active_export_jobs_returns_queue_state(self):
        queued_job = ExportJob(
            id="front_door_queued",
            camera="front_door",
            status="queued",
            request_start_time=100,
            request_end_time=150,
        )

        with patch(
            "frigate.api.export.list_active_export_jobs",
            return_value=[queued_job],
        ):
            with AuthTestClient(self.app) as client:
                response = client.get("/jobs/export")

        assert response.status_code == 200
        assert response.json() == [queued_job.to_dict()]

    def test_rename_export_moves_the_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "front_door_20260823_020615_abc123.mp4")
            thumb = os.path.join(tmpdir, "front_door_abc123.webp")
            for path, data in ((video, b"video"), (thumb, b"thumb")):
                with open(path, "wb") as handle:
                    handle.write(data)

            Export.create(
                id="front_door_abc123",
                camera="front_door",
                name="front door 2026-08-23 02:06:15 2026-08-23 02:07:34",
                date=100,
                video_path=video,
                thumb_path=thumb,
                in_progress=False,
            )

            with patch("frigate.record.export.EXPORT_DIR", tmpdir):
                with AuthTestClient(self.app) as client:
                    response = client.patch(
                        "/export/front_door_abc123/rename",
                        json={"name": "Package thief"},
                    )

            assert response.status_code == 200

            renamed = Export.get(Export.id == "front_door_abc123")
            assert renamed.name == "Package thief"
            assert os.path.basename(renamed.video_path) == "Package thief_abc123.mp4"
            assert os.path.exists(renamed.video_path)
            assert not os.path.exists(video)

    def test_rename_export_rejected_while_in_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "front_door_abc123.mp4")
            with open(video, "wb") as handle:
                handle.write(b"video")

            Export.create(
                id="front_door_running",
                camera="front_door",
                name="front door export",
                date=100,
                video_path=video,
                thumb_path=os.path.join(tmpdir, "t.webp"),
                in_progress=True,
            )

            with AuthTestClient(self.app) as client:
                response = client.patch(
                    "/export/front_door_running/rename",
                    json={"name": "Package thief"},
                )

            assert response.status_code == 400
            assert Export.get(Export.id == "front_door_running").video_path == video

    def test_delete_export_rejected_while_in_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "front_door_abc123.mp4")
            with open(video, "wb") as handle:
                handle.write(b"video")

            Export.create(
                id="front_door_running",
                camera="front_door",
                name="front door export",
                date=100,
                video_path=video,
                thumb_path=os.path.join(tmpdir, "t.webp"),
                in_progress=True,
            )

            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/delete", json={"ids": ["front_door_running"]}
                )

            assert response.status_code == 400
            assert os.path.exists(video)

    def test_rename_export_missing_file_leaves_the_row_alone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "front_door_gone_abc123.mp4")

            Export.create(
                id="front_door_gone",
                camera="front_door",
                name="front door export",
                date=100,
                video_path=video,
                thumb_path=os.path.join(tmpdir, "t.webp"),
                in_progress=False,
            )

            with patch("frigate.record.export.EXPORT_DIR", tmpdir):
                with AuthTestClient(self.app) as client:
                    response = client.patch(
                        "/export/front_door_gone/rename",
                        json={"name": "Package thief"},
                    )

            assert response.status_code == 500

            unchanged = Export.get(Export.id == "front_door_gone")
            assert unchanged.name == "front door export"
            assert unchanged.video_path == video

    def test_reap_stale_exports_removes_staged_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging_dir = os.path.join(tmpdir, "staging")
            os.makedirs(staging_dir)
            staged = os.path.join(staging_dir, "front_door_abc_0.mp4")
            finished = os.path.join(tmpdir, "export_stage_saved_abc.mp4")

            for path in (staged, finished):
                with open(path, "w") as handle:
                    handle.write("video")

            with patch("frigate.jobs.export.EXPORT_STAGING_DIR", staging_dir):
                reap_stale_exports()

            assert not os.path.exists(staging_dir)
            assert os.path.exists(finished)

    def test_reap_stale_exports_deletes_rows_with_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stale_video = os.path.join(tmpdir, "stale.mp4")
            stale_thumb = os.path.join(tmpdir, "stale.webp")
            # stale_video is intentionally NOT created
            with open(stale_thumb, "w") as handle:
                handle.write("thumb")

            Export.create(
                id="stale_no_file",
                camera="front_door",
                name="Stuck export",
                date=100,
                video_path=stale_video,
                thumb_path=stale_thumb,
                in_progress=True,
            )

            reap_stale_exports()

            assert Export.get_or_none(Export.id == "stale_no_file") is None
            assert not os.path.exists(stale_thumb)

    def test_reap_stale_exports_recovers_rows_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intact_video = os.path.join(tmpdir, "intact.mp4")
            intact_thumb = os.path.join(tmpdir, "intact.webp")
            with open(intact_video, "wb") as handle:
                handle.write(b"not actually an mp4 but non-empty")
            with open(intact_thumb, "wb") as handle:
                handle.write(b"thumb")

            case = ExportCase.create(
                id="case_for_stale",
                name="Curated case",
                description="",
                created_at=10,
                updated_at=10,
            )

            Export.create(
                id="stale_with_file",
                camera="front_door",
                name="Recoverable export",
                date=200,
                video_path=intact_video,
                thumb_path=intact_thumb,
                in_progress=True,
                export_case=case,
            )

            reap_stale_exports()

            recovered = Export.get(Export.id == "stale_with_file")
            assert recovered.in_progress is False
            # Case link must be cleared so the user re-triages the recovered row
            assert recovered.export_case is None
            # The case itself is untouched
            assert ExportCase.get_or_none(ExportCase.id == "case_for_stale") is not None
            # Recovered files must NOT be unlinked
            assert os.path.exists(intact_video)
            assert os.path.exists(intact_thumb)

    def test_reap_stale_exports_delete_path_severs_case_link(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_video = os.path.join(tmpdir, "missing.mp4")
            # file intentionally not created

            case = ExportCase.create(
                id="case_losing_member",
                name="Case losing a member",
                description="",
                created_at=20,
                updated_at=20,
            )

            Export.create(
                id="stale_in_case_no_file",
                camera="front_door",
                name="Stuck and in a case",
                date=250,
                video_path=missing_video,
                thumb_path="",
                in_progress=True,
                export_case=case,
            )

            reap_stale_exports()

            # The export row is gone entirely
            assert Export.get_or_none(Export.id == "stale_in_case_no_file") is None
            # The case stays but has no exports pointing at it
            remaining_case = ExportCase.get(ExportCase.id == "case_losing_member")
            assert list(remaining_case.exports) == []

    def test_reap_stale_exports_deletes_rows_with_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_video = os.path.join(tmpdir, "empty.mp4")
            # Create a zero-byte file — partial ffmpeg output
            open(empty_video, "w").close()

            Export.create(
                id="stale_empty_file",
                camera="front_door",
                name="Zero byte export",
                date=300,
                video_path=empty_video,
                thumb_path="",
                in_progress=True,
            )

            reap_stale_exports()

            assert Export.get_or_none(Export.id == "stale_empty_file") is None
            assert not os.path.exists(empty_video)

    def test_reap_stale_exports_skips_completed_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            done_video = os.path.join(tmpdir, "done.mp4")
            with open(done_video, "wb") as handle:
                handle.write(b"done")

            Export.create(
                id="already_done",
                camera="front_door",
                name="Completed export",
                date=400,
                video_path=done_video,
                thumb_path="",
                in_progress=False,
            )

            reap_stale_exports()

            row = Export.get(Export.id == "already_done")
            assert row.in_progress is False
            assert os.path.exists(done_video)

    def test_batch_export_without_case_goes_to_uncategorized(self):
        """Exports without a case target go to uncategorized."""
        self._insert_recording("rec-front", "front_door", 100, 400)

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            }
                        ],
                    },
                )

        assert response.status_code == 202
        response_json = response.json()
        assert response_json["export_case_id"] is None
        assert ExportCase.select().count() == 0

    # --- /exports/batch (item-shaped multi-export) ---------------------------

    def test_batch_export_happy_path_creates_case_and_queues_all(self):
        self._insert_recording("rec-front", "front_door", 100, 400)
        self._insert_recording("rec-back", "backyard", 100, 400)

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            },
                            {
                                "camera": "front_door",
                                "start_time": 200,
                                "end_time": 240,
                            },
                            {
                                "camera": "backyard",
                                "start_time": 300,
                                "end_time": 340,
                            },
                        ],
                        "new_case_name": "Incident Apr 11",
                        "new_case_description": "Review items",
                    },
                )

        assert response.status_code == 202
        response_json = response.json()
        assert len(response_json["export_ids"]) == 3
        assert all(r["success"] for r in response_json["results"])
        assert [r["item_index"] for r in response_json["results"]] == [0, 1, 2]
        assert start_export_job.call_count == 3

        case = ExportCase.get(ExportCase.id == response_json["export_case_id"])
        assert case.name == "Incident Apr 11"
        assert case.description == "Review items"

    def test_batch_export_existing_case_does_not_create_new_case(self):
        self._insert_recording("rec-front", "front_door", 100, 400)
        ExportCase.create(
            id="existing_case",
            name="Existing",
            description="",
            created_at=10,
            updated_at=10,
        )

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            }
                        ],
                        "export_case_id": "existing_case",
                    },
                )

        assert response.status_code == 202
        assert response.json()["export_case_id"] == "existing_case"
        # No additional case was created
        assert ExportCase.select().count() == 1

    def test_batch_export_empty_items_rejected(self):
        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/batch",
                json={"items": [], "new_case_name": "Empty"},
            )

        assert response.status_code == 422

    def test_batch_export_over_limit_rejected(self):
        items = [
            {"camera": "front_door", "start_time": 100 + i, "end_time": 100 + i + 5}
            for i in range(51)
        ]

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/batch",
                json={"items": items, "new_case_name": "Too many"},
            )

        assert response.status_code == 422

    def test_batch_export_end_before_start_rejected(self):
        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/batch",
                json={
                    "items": [
                        {
                            "camera": "front_door",
                            "start_time": 200,
                            "end_time": 100,
                        }
                    ],
                    "new_case_name": "Bad range",
                },
            )

        assert response.status_code == 422
        assert (
            response.json()["detail"][0]["msg"]
            == "Value error, end_time must be after start_time"
        )

    def test_batch_export_non_admin_without_case_goes_to_uncategorized(self):
        """Non-admin batch exports go to uncategorized."""
        self._insert_recording("rec-front", "front_door", 100, 400)

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    headers={"remote-user": "viewer", "remote-role": "viewer"},
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 100,
                                "end_time": 150,
                            }
                        ],
                    },
                )

        assert response.status_code == 202
        response_json = response.json()
        assert response_json["export_case_id"] is None
        assert ExportCase.select().count() == 0

    def test_batch_export_camera_access_denied_fails_closed(self):
        from fastapi import Request

        from frigate.api.auth import get_allowed_cameras_for_filter

        self._insert_recording("rec-front", "front_door", 100, 400)

        async def restricted(request: Request):
            return ["front_door"]

        self.app.dependency_overrides[get_allowed_cameras_for_filter] = restricted

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            },
                            {
                                "camera": "backyard",  # not in allowed list
                                "start_time": 110,
                                "end_time": 150,
                            },
                        ],
                        "new_case_name": "Nope",
                    },
                )

        assert response.status_code == 403
        start_export_job.assert_not_called()
        # No case created
        assert ExportCase.select().count() == 0

    def test_batch_export_case_not_found(self):
        self._insert_recording("rec-front", "front_door", 100, 400)

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/batch",
                json={
                    "items": [
                        {
                            "camera": "front_door",
                            "start_time": 110,
                            "end_time": 150,
                        }
                    ],
                    "export_case_id": "does_not_exist",
                },
            )

        assert response.status_code == 404

    def test_batch_export_per_item_missing_recordings_partial_success(self):
        self._insert_recording("rec-front", "front_door", 100, 200)
        # backyard has no recordings at all

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            },
                            {
                                "camera": "backyard",
                                "start_time": 110,
                                "end_time": 150,
                            },
                        ],
                        "new_case_name": "Partial",
                    },
                )

        assert response.status_code == 202
        response_json = response.json()
        assert len(response_json["export_ids"]) == 1
        results_by_camera = {r["camera"]: r for r in response_json["results"]}
        assert results_by_camera["front_door"]["success"] is True
        assert results_by_camera["backyard"]["success"] is False
        assert (
            results_by_camera["backyard"]["error"]
            == "No recordings found for time range"
        )
        start_export_job.assert_called_once()

        # Case is still created because at least one item succeeded
        assert (
            ExportCase.get(ExportCase.id == response_json["export_case_id"]) is not None
        )

    def test_batch_export_same_camera_different_ranges_one_missing(self):
        # Recording covers 100-200 only. First item fits, second does not.
        self._insert_recording("rec-front", "front_door", 100, 200)

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            },
                            {
                                "camera": "front_door",
                                "start_time": 500,
                                "end_time": 540,
                            },
                        ],
                        "new_case_name": "Split recordings",
                    },
                )

        assert response.status_code == 202
        response_json = response.json()
        assert len(response_json["export_ids"]) == 1
        results = response_json["results"]
        assert results[0]["success"] is True
        assert results[0]["item_index"] == 0
        assert results[1]["success"] is False
        assert results[1]["item_index"] == 1
        assert results[1]["error"] == "No recordings found for time range"
        # Both results carry the same camera — item_index is the only way
        # the client can tell them apart.
        assert results[0]["camera"] == results[1]["camera"] == "front_door"
        start_export_job.assert_called_once()

    def test_batch_export_all_missing_recordings_rolls_back_case(self):
        # No recordings inserted at all
        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            }
                        ],
                        "new_case_name": "Should rollback",
                    },
                )

        assert response.status_code == 400
        start_export_job.assert_not_called()
        assert ExportCase.select().count() == 0

    def test_batch_export_preflight_queue_full(self):
        self._insert_recording("rec-front", "front_door", 100, 400)
        self._insert_recording("rec-back", "backyard", 100, 400)

        with patch(
            "frigate.api.export.available_export_queue_slots",
            return_value=1,
        ):
            with patch(
                "frigate.api.export.start_export_job",
                side_effect=lambda _config, job: job.id,
            ) as start_export_job:
                with AuthTestClient(self.app) as client:
                    response = client.post(
                        "/exports/batch",
                        json={
                            "items": [
                                {
                                    "camera": "front_door",
                                    "start_time": 110,
                                    "end_time": 150,
                                },
                                {
                                    "camera": "backyard",
                                    "start_time": 110,
                                    "end_time": 150,
                                },
                            ],
                            "new_case_name": "Queue full",
                        },
                    )

        assert response.status_code == 503
        start_export_job.assert_not_called()
        assert ExportCase.select().count() == 0

    def test_batch_export_all_enqueue_calls_fail_rolls_back_case(self):
        self._insert_recording("rec-front", "front_door", 100, 400)

        def boom(_config, _job):
            raise RuntimeError("simulated enqueue failure")

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=boom,
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            }
                        ],
                        "new_case_name": "Will fail",
                    },
                )

        assert response.status_code == 202
        response_json = response.json()
        assert response_json["export_ids"] == []
        assert response_json["export_case_id"] is None
        assert ExportCase.select().count() == 0

    def test_batch_export_rejects_invalid_image_path(self):
        self._insert_recording("rec-front", "front_door", 100, 400)

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/batch",
                json={
                    "items": [
                        {
                            "camera": "front_door",
                            "start_time": 110,
                            "end_time": 150,
                            "image_path": "/etc/passwd",
                        }
                    ],
                    "new_case_name": "Bad image",
                },
            )

        assert response.status_code == 400
        assert ExportCase.select().count() == 0

    def _batch_export_with_image(self, image_path: str, role: str):
        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    headers={"remote-user": role, "remote-role": role},
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                                "image_path": image_path,
                            }
                        ],
                    },
                )

        return response, start_export_job

    def test_batch_export_restricted_role_rejects_unreadable_image_path(self):
        self._insert_recording("rec-front", "front_door", 100, 400)
        self.app.frigate_config.auth.roles["limited_user"] = ["front_door"]

        for image_path in (
            f"{CLIPS_DIR}/review/thumb-backyard-123.456-abc.webp",
            f"{CLIPS_DIR}/thumbs/backyard/123.webp",
            f"{CLIPS_DIR}/faces/someone/1.webp",
            f"{CLIPS_DIR}/custom/thumb.jpg",
        ):
            response, _ = self._batch_export_with_image(image_path, "limited_user")

            assert response.status_code == 400, image_path

    def test_batch_export_restricted_role_accepts_own_camera_image_path(self):
        self._insert_recording("rec-front", "front_door", 100, 400)
        self.app.frigate_config.auth.roles["limited_user"] = ["front_door"]
        image_path = f"{CLIPS_DIR}/review/thumb-front_door-123.456-abc.webp"

        response, start_export_job = self._batch_export_with_image(
            image_path, "limited_user"
        )

        assert response.status_code == 202
        assert start_export_job.call_args.args[1].image_path == image_path

    def test_batch_export_unrestricted_roles_accept_custom_image_path(self):
        self._insert_recording("rec-front", "front_door", 100, 400)
        image_path = f"{CLIPS_DIR}/custom/thumb.jpg"

        for role in ("admin", "viewer"):
            response, start_export_job = self._batch_export_with_image(image_path, role)

            assert response.status_code == 202, role
            assert start_export_job.call_args.args[1].image_path == image_path

    def test_batch_export_non_admin_can_queue(self):
        self._insert_recording("rec-front", "front_door", 100, 400)

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    headers={"remote-user": "viewer", "remote-role": "viewer"},
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            }
                        ],
                        "new_case_name": "Viewer export",
                    },
                )

        assert response.status_code == 202
        assert len(response.json()["export_ids"]) == 1

    def test_batch_export_non_admin_cannot_attach_to_existing_case(self):
        """Non-admins can create cases via new_case_name but cannot attach
        to existing cases they did not create. Closes a write-path hole that
        would otherwise be reachable through the unfiltered GET /cases list.
        """
        self._insert_recording("rec-front", "front_door", 100, 400)
        ExportCase.create(
            id="admins_only_case",
            name="Admins only",
            description="",
            created_at=10,
            updated_at=10,
        )

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    headers={"remote-user": "viewer", "remote-role": "viewer"},
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            }
                        ],
                        "export_case_id": "admins_only_case",
                    },
                )

        assert response.status_code == 403
        start_export_job.assert_not_called()
        # No exports should have been created in the target case
        assert Export.select().count() == 0

    def test_batch_export_admin_can_attach_to_existing_case(self):
        self._insert_recording("rec-front", "front_door", 100, 400)
        ExportCase.create(
            id="shared_case",
            name="Shared",
            description="",
            created_at=10,
            updated_at=10,
        )

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                            }
                        ],
                        "export_case_id": "shared_case",
                    },
                )

        assert response.status_code == 202
        assert response.json()["export_case_id"] == "shared_case"
        # No additional case created
        assert ExportCase.select().count() == 1

    def test_batch_export_roundtrips_client_item_id(self):
        self._insert_recording("rec-front", "front_door", 100, 400)

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/batch",
                    json={
                        "items": [
                            {
                                "camera": "front_door",
                                "start_time": 110,
                                "end_time": 150,
                                "client_item_id": "review-123",
                            }
                        ],
                        "new_case_name": "Client id test",
                    },
                )

        assert response.status_code == 202
        assert response.json()["results"][0]["client_item_id"] == "review-123"

    def test_single_export_non_admin_cannot_attach_to_existing_case(self):
        """The single-export route has the same hole: non-admins should not
        be able to smuggle exports into an existing case via export_case_id.
        Admin-gating this matches /exports/batch.
        """
        self._insert_recording("rec-front", "front_door", 100, 400)
        ExportCase.create(
            id="admins_only_case",
            name="Admins only",
            description="",
            created_at=10,
            updated_at=10,
        )

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ) as start_export_job:
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/export/front_door/start/110/end/150",
                    headers={"remote-user": "viewer", "remote-role": "viewer"},
                    json={"export_case_id": "admins_only_case"},
                )

        assert response.status_code == 403
        start_export_job.assert_not_called()
        assert Export.select().count() == 0

    def test_single_export_non_admin_can_still_export_without_case(self):
        """Regression guard: the admin gate only applies to export_case_id,
        not to single exports in general. Non-admins should still be able
        to start a single export for a camera they have access to.
        """
        self._insert_recording("rec-front", "front_door", 100, 400)

        with patch(
            "frigate.api.export.start_export_job",
            side_effect=lambda _config, job: job.id,
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/export/front_door/start/110/end/150",
                    headers={"remote-user": "viewer", "remote-role": "viewer"},
                    json={},
                )

        assert response.status_code == 202
        assert response.json()["success"] is True

    # ── Bulk delete exports ────────────────────────────────────────

    def test_bulk_delete_exports_success(self):
        """All IDs exist, none in-progress → 200, all deleted."""
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
        )
        Export.create(
            id="exp2",
            camera="front_door",
            name="export_2",
            date=200,
            video_path="/tmp/exp2.mp4",
            thumb_path="/tmp/exp2.jpg",
            in_progress=False,
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/delete",
                json={"ids": ["exp1", "exp2"]},
            )

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert Export.select().count() == 0

    def test_bulk_delete_exports_single_item(self):
        """Regression: single-item delete via batch endpoint."""
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/delete",
                json={"ids": ["exp1"]},
            )

        assert response.status_code == 200
        assert Export.select().count() == 0

    def test_bulk_delete_exports_some_missing(self):
        """Some IDs don't exist → 404, nothing deleted."""
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/delete",
                json={"ids": ["exp1", "nonexistent"]},
            )

        assert response.status_code == 404
        # Nothing deleted
        assert Export.select().count() == 1

    def test_bulk_delete_exports_all_missing(self):
        """All IDs don't exist → 404."""
        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/delete",
                json={"ids": ["nope1", "nope2"]},
            )

        assert response.status_code == 404

    def test_bulk_delete_exports_in_progress(self):
        """Some exports in-progress → 400, nothing deleted."""
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path=f"{os.environ.get('EXPORT_DIR', '/media/frigate/exports')}/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=True,
        )

        with patch(
            "frigate.api.export._get_files_in_use",
            return_value={"exp1.mp4"},
        ):
            with AuthTestClient(self.app) as client:
                response = client.post(
                    "/exports/delete",
                    json={"ids": ["exp1"]},
                )

        assert response.status_code == 400
        assert Export.select().count() == 1

    def test_bulk_delete_exports_non_admin_rejected(self):
        """Non-admin users cannot bulk delete."""
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/delete",
                headers={"remote-user": "viewer", "remote-role": "viewer"},
                json={"ids": ["exp1"]},
            )

        assert response.status_code == 403
        assert Export.select().count() == 1

    # ── Bulk reassign exports ──────────────────────────────────────

    def test_bulk_reassign_exports_to_case(self):
        """All IDs exist, case exists → 200, all reassigned."""
        ExportCase.create(
            id="case1",
            name="Test Case",
            description="",
            created_at=10,
            updated_at=10,
        )
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
        )
        Export.create(
            id="exp2",
            camera="front_door",
            name="export_2",
            date=200,
            video_path="/tmp/exp2.mp4",
            thumb_path="/tmp/exp2.jpg",
            in_progress=False,
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/reassign",
                json={"ids": ["exp1", "exp2"], "export_case_id": "case1"},
            )

        assert response.status_code == 200
        assert response.json()["success"] is True
        for exp_id in ["exp1", "exp2"]:
            exp = Export.get(Export.id == exp_id)
            assert exp.export_case_id == "case1"

    def test_bulk_reassign_exports_to_null(self):
        """Reassign to null (uncategorize) → 200."""
        ExportCase.create(
            id="case1",
            name="Test Case",
            description="",
            created_at=10,
            updated_at=10,
        )
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
            export_case="case1",
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/reassign",
                json={"ids": ["exp1"], "export_case_id": None},
            )

        assert response.status_code == 200
        exp = Export.get(Export.id == "exp1")
        assert exp.export_case_id is None

    def test_bulk_reassign_exports_single_item(self):
        """Regression: single-item reassign via batch endpoint."""
        ExportCase.create(
            id="case1",
            name="Test Case",
            description="",
            created_at=10,
            updated_at=10,
        )
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/reassign",
                json={"ids": ["exp1"], "export_case_id": "case1"},
            )

        assert response.status_code == 200
        exp = Export.get(Export.id == "exp1")
        assert exp.export_case_id == "case1"

    def test_bulk_reassign_exports_some_missing(self):
        """Some IDs don't exist → 404, nothing reassigned."""
        ExportCase.create(
            id="case1",
            name="Test Case",
            description="",
            created_at=10,
            updated_at=10,
        )
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/reassign",
                json={
                    "ids": ["exp1", "nonexistent"],
                    "export_case_id": "case1",
                },
            )

        assert response.status_code == 404
        # Nothing reassigned
        exp = Export.get(Export.id == "exp1")
        assert exp.export_case_id is None

    def test_bulk_reassign_exports_case_not_found(self):
        """Target case doesn't exist → 404."""
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/reassign",
                json={"ids": ["exp1"], "export_case_id": "nonexistent"},
            )

        assert response.status_code == 404
        exp = Export.get(Export.id == "exp1")
        assert exp.export_case_id is None

    def test_bulk_reassign_exports_non_admin_rejected(self):
        """Non-admin users cannot bulk reassign."""
        Export.create(
            id="exp1",
            camera="front_door",
            name="export_1",
            date=100,
            video_path="/tmp/exp1.mp4",
            thumb_path="/tmp/exp1.jpg",
            in_progress=False,
        )

        with AuthTestClient(self.app) as client:
            response = client.post(
                "/exports/reassign",
                headers={"remote-user": "viewer", "remote-role": "viewer"},
                json={"ids": ["exp1"], "export_case_id": None},
            )

        assert response.status_code == 403

    def test_download_export_case_with_multibyte_name(self):
        """A case name outside latin-1 must not break the response headers."""
        case = ExportCase.create(
            id="case_multibyte",
            name="テスト事案",
            description="",
            created_at=10,
            updated_at=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "現場カメラ.mp4")
            with open(video_path, "wb") as handle:
                handle.write(b"video")

            Export.create(
                id="export_multibyte",
                camera="front_door",
                name="現場カメラ",
                date=100,
                video_path=video_path,
                thumb_path=os.path.join(tmpdir, "multibyte_export.webp"),
                in_progress=False,
                export_case=case,
            )

            with AuthTestClient(self.app) as client:
                response = client.get(f"/cases/{case.id}/download")

        assert response.status_code == 200
        # RFC 5987/6266: the UTF-8 name rides in filename*, and a latin-1 safe
        # fallback stays in filename for old clients.
        assert response.headers["content-disposition"] == (
            'attachment; filename="case_multibyte.zip"; '
            "filename*=UTF-8''%E3%83%86%E3%82%B9%E3%83%88%E4%BA%8B%E6%A1%88.zip"
        )

        archive = zipfile.ZipFile(io.BytesIO(response.content))
        assert archive.namelist() == ["現場カメラ.mp4"]

    def test_download_export_case_with_ascii_name(self):
        """An ASCII case name still gets a plain, readable filename."""
        case = ExportCase.create(
            id="case_ascii",
            name="Burglary 2026-08",
            description="",
            created_at=10,
            updated_at=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "ascii_export.mp4")
            with open(video_path, "wb") as handle:
                handle.write(b"video")

            Export.create(
                id="export_ascii",
                camera="front_door",
                name="Front door",
                date=100,
                video_path=video_path,
                thumb_path=os.path.join(tmpdir, "ascii_export.webp"),
                in_progress=False,
                export_case=case,
            )

            with AuthTestClient(self.app) as client:
                response = client.get(f"/cases/{case.id}/download")

        assert response.status_code == 200
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="Burglary 2026-08.zip"; '
            "filename*=UTF-8''Burglary%202026-08.zip"
        )

    def _create_video_case(
        self, tmpdir: str, exports: list[tuple[str, str, str, float]]
    ) -> ExportCase:
        """Create a case holding one export per (id, camera, name, date)."""
        case = ExportCase.create(
            id="case_video",
            name="Front door incident",
            description="",
            created_at=10,
            updated_at=10,
        )

        for export_id, camera, name, date in exports:
            video_path = os.path.join(tmpdir, f"{export_id}.mp4")
            with open(video_path, "wb") as handle:
                handle.write(b"video")

            Export.create(
                id=export_id,
                camera=camera,
                name=name,
                date=date,
                video_path=video_path,
                thumb_path=os.path.join(tmpdir, f"{export_id}.webp"),
                in_progress=False,
                export_case=case,
            )

        return case

    @staticmethod
    def _fake_probe(signatures: dict[str, tuple] | None = None):
        """Stand in for ffprobe: 2.5s per export, one shared format by default."""
        from frigate.api.export import _CaseVideoSource

        def probe(_ffprobe_path: str, export: Export) -> _CaseVideoSource:
            signature = (signatures or {}).get(export.id, (("h264", 1920, 1080), ()))
            return _CaseVideoSource(export=export, duration=2.5, signature=signature)

        return probe

    def test_download_export_case_as_video_joins_exports_in_start_order(self):
        captured: dict[str, object] = {}

        def fake_stream(ffmpeg_cmd: list[str], playlist_path: str):
            chapters_path = ffmpeg_cmd[ffmpeg_cmd.index("ffmetadata") + 2]
            captured["cmd"] = ffmpeg_cmd
            captured["work_dir"] = os.path.dirname(playlist_path)
            with open(playlist_path) as playlist:
                captured["playlist"] = playlist.read()
            with open(chapters_path) as chapters:
                captured["chapters"] = chapters.read()
            yield b"joined video"

        with tempfile.TemporaryDirectory() as tmpdir:
            case = self._create_video_case(
                tmpdir,
                [
                    ("export_late", "front_door", "Late; side=door", 200),
                    ("export_early", "front_door", "Early", 100),
                ],
            )

            with (
                patch("frigate.api.export.CACHE_DIR", tmpdir),
                patch(
                    "frigate.api.export._probe_case_video_source",
                    side_effect=self._fake_probe(),
                ),
                patch("frigate.api.export._run_clip_download", fake_stream),
            ):
                with AuthTestClient(self.app) as client:
                    response = client.get(f"/cases/{case.id}/download?format=mp4")

            assert response.status_code == 200
            assert response.content == b"joined video"
            assert response.headers["content-type"] == "video/mp4"
            assert response.headers["content-disposition"] == (
                'attachment; filename="Front door incident.mp4"; '
                "filename*=UTF-8''Front%20door%20incident.mp4"
            )

            assert captured["playlist"] == (
                f"file '{os.path.join(tmpdir, 'export_early.mp4')}'\n"
                f"file '{os.path.join(tmpdir, 'export_late.mp4')}'\n"
            )
            # one chapter per export, back to back in output time
            assert captured["chapters"] == (
                ";FFMETADATA1\n"
                "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2500\ntitle=Early\n"
                "[CHAPTER]\nTIMEBASE=1/1000\nSTART=2500\nEND=5000\n"
                "title=Late\\; side\\=door\n"
            )

            cmd = captured["cmd"]
            assert cmd[cmd.index("-c") + 1] == "copy"
            assert ["-map", "0:v:0", "-map", "0:a:0?"] == cmd[
                cmd.index("-map") : cmd.index("-map") + 4
            ]
            assert cmd[cmd.index("-map_metadata") + 1] == "1"
            assert cmd[cmd.index("-map_chapters") + 1] == "1"
            assert not os.path.exists(captured["work_dir"])

    def test_download_export_case_as_video_rejects_mixed_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case = self._create_video_case(
                tmpdir,
                [
                    ("export_front", "front_door", "Front", 100),
                    ("export_back", "backyard", "Back", 200),
                ],
            )

            with (
                patch("frigate.api.export.CACHE_DIR", tmpdir),
                patch(
                    "frigate.api.export._probe_case_video_source",
                    side_effect=self._fake_probe(
                        {"export_back": (("h264", 640, 480), ())}
                    ),
                ),
                patch("frigate.api.export._run_clip_download") as stream,
            ):
                with AuthTestClient(self.app) as client:
                    response = client.get(f"/cases/{case.id}/download?format=mp4")

            assert response.status_code == 400
            assert "zip" in response.json()["message"]
            stream.assert_not_called()
            assert not [p for p in os.listdir(tmpdir) if p.startswith("case_video_")]

    def test_download_export_case_as_video_joins_matching_cameras(self):
        """Cameras with the same format are joined, ordered by start then camera."""
        playlists: list[str] = []

        def fake_stream(_ffmpeg_cmd: list[str], playlist_path: str):
            with open(playlist_path) as playlist:
                playlists.append(playlist.read())
            yield b"video"

        with tempfile.TemporaryDirectory() as tmpdir:
            case = self._create_video_case(
                tmpdir,
                [
                    ("export_front", "front_door", "Front", 100),
                    ("export_back", "backyard", "Back", 100),
                ],
            )

            with (
                patch("frigate.api.export.CACHE_DIR", tmpdir),
                patch(
                    "frigate.api.export._probe_case_video_source",
                    side_effect=self._fake_probe(),
                ),
                patch("frigate.api.export._run_clip_download", fake_stream),
            ):
                with AuthTestClient(self.app) as client:
                    response = client.get(f"/cases/{case.id}/download?format=mp4")

            assert response.status_code == 200
            assert playlists == [
                f"file '{os.path.join(tmpdir, 'export_back.mp4')}'\n"
                f"file '{os.path.join(tmpdir, 'export_front.mp4')}'\n"
            ]

    def test_download_export_case_as_video_fails_on_unreadable_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case = self._create_video_case(
                tmpdir, [("export_broken", "front_door", "Broken", 100)]
            )

            with (
                patch("frigate.api.export._probe_case_video_source", return_value=None),
                patch("frigate.api.export._run_clip_download") as stream,
            ):
                with AuthTestClient(self.app) as client:
                    response = client.get(f"/cases/{case.id}/download?format=mp4")

        assert response.status_code == 500
        stream.assert_not_called()

    def test_download_export_case_as_video_only_includes_allowed_cameras(self):
        from fastapi import Request

        from frigate.api.auth import get_allowed_cameras_for_filter

        async def restricted(request: Request):
            return ["front_door"]

        self.app.dependency_overrides[get_allowed_cameras_for_filter] = restricted
        playlists: list[str] = []

        def fake_stream(_ffmpeg_cmd: list[str], playlist_path: str):
            with open(playlist_path) as playlist:
                playlists.append(playlist.read())
            yield b"video"

        with tempfile.TemporaryDirectory() as tmpdir:
            case = self._create_video_case(
                tmpdir,
                [
                    ("export_front", "front_door", "Front", 100),
                    ("export_back", "backyard", "Back", 200),
                ],
            )

            with (
                patch("frigate.api.export.CACHE_DIR", tmpdir),
                patch(
                    "frigate.api.export._probe_case_video_source",
                    side_effect=self._fake_probe(),
                ),
                patch("frigate.api.export._run_clip_download", fake_stream),
            ):
                with AuthTestClient(self.app) as client:
                    response = client.get(f"/cases/{case.id}/download?format=mp4")

            assert response.status_code == 200
            assert playlists == [f"file '{os.path.join(tmpdir, 'export_front.mp4')}'\n"]

    def test_download_export_case_as_video_without_files_is_not_found(self):
        case = ExportCase.create(
            id="case_missing",
            name="Missing",
            description="",
            created_at=10,
            updated_at=10,
        )
        Export.create(
            id="export_missing",
            camera="front_door",
            name="Missing",
            date=100,
            video_path="/nonexistent/export_missing.mp4",
            thumb_path="/nonexistent/export_missing.webp",
            in_progress=False,
            export_case=case,
        )

        with patch("frigate.api.export._probe_case_video_source") as probe:
            with AuthTestClient(self.app) as client:
                response = client.get(f"/cases/{case.id}/download?format=mp4")

        assert response.status_code == 404
        probe.assert_not_called()

    def test_download_export_case_rejects_unknown_format(self):
        with AuthTestClient(self.app) as client:
            response = client.get("/cases/any/download?format=avi")

        assert response.status_code == 422

    def test_probe_case_video_source_signature(self):
        from frigate.api.export import _probe_case_video_source

        def ffprobe_output(extradata: str) -> str:
            return json.dumps(
                {
                    "streams": [
                        {
                            "index": 0,
                            "codec_type": "video",
                            "codec_name": "h264",
                            "profile": "High",
                            "width": 2560,
                            "height": 1440,
                            "pix_fmt": "yuv420p",
                            "extradata_hash": extradata,
                        },
                        {"index": 1, "codec_type": "data"},
                    ],
                    "format": {"duration": "65.978000"},
                }
            )

        export = Export(id="export_probe", video_path="/media/export.mp4")

        def probe(stdout: str, returncode: int = 0):
            completed = sp.CompletedProcess([], returncode, stdout.encode(), b"")
            with patch("frigate.api.export.sp.run", return_value=completed) as run:
                source = _probe_case_video_source("ffprobe", export)
            return source, run

        first, run = probe(ffprobe_output("SHA256:aaaa"))
        same, _ = probe(ffprobe_output("SHA256:aaaa"))
        other_sps, _ = probe(ffprobe_output("SHA256:bbbb"))

        assert run.call_args.args[0][-1] == "/media/export.mp4"
        assert first is not None
        assert first.duration == 65.978
        assert first.signature == same.signature
        assert first.signature != other_sps.signature

        failed, _ = probe("", returncode=1)
        assert failed is None

        no_video, _ = probe(json.dumps({"streams": [], "format": {"duration": "1"}}))
        assert no_video is None
