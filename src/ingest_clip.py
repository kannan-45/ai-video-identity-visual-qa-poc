import hashlib
import json
from pathlib import Path


class ClipIngestor:
    """
    Validates and ingests a video clip together with
    its formal ClipMetadata contract.
    """

    REQUIRED_FIELDS = {
        "clip_id",
        "shot_id",
        "mode",
        "source_type",
        "file",
        "duration_s",
        "fps",
        "width",
        "height",
        "aspect_ratio",
        "codec",
        "model",
        "settings",
        "reference_ids",
        "motion_controls",
        "seed",
        "status",
        "failure_reason",
        "checksum_sha256",
    }

    def _load_metadata(self, metadata_path):
        metadata_path = Path(metadata_path)

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Clip metadata not found: {metadata_path}"
            )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        missing = sorted(
            self.REQUIRED_FIELDS - set(metadata)
        )

        if missing:
            raise ValueError(
                "ClipMetadata is missing required fields: "
                f"{missing}"
            )

        return metadata

    @staticmethod
    def _calculate_sha256(video_path):
        sha256 = hashlib.sha256()

        with open(video_path, "rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    def ingest_clip(
        self,
        video_path,
        metadata_path,
    ):
        video_path = Path(video_path).resolve()
        metadata_path = Path(metadata_path).resolve()

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        metadata = self._load_metadata(
            metadata_path
        )

        metadata_root = metadata_path.parents[3]

        metadata_video = (
            metadata_root
            / Path(metadata["file"])
        ).resolve()

        if metadata_video != video_path:
            raise ValueError(
                "ClipMetadata file does not match "
                f"the supplied video.\n"
                f"Metadata: {metadata_video}\n"
                f"Video: {video_path}"
            )

        actual_checksum = (
            self._calculate_sha256(video_path)
        )

        checksum_verified = (
            actual_checksum.lower()
            == metadata["checksum_sha256"].lower()
        )

        if not checksum_verified:
            raise ValueError(
                "Clip checksum does not match "
                "ClipMetadata."
            )

        return {
            "clip_id": metadata["clip_id"],
            "video_path": str(video_path),
            "metadata_path": str(metadata_path),
            "metadata": metadata,
            "checksum_verified": True,
            "ingest_status": "SUCCESS",
        }