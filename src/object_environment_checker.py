import cv2
import numpy as np


class ObjectEnvironmentChecker:

    def __init__(self):
        pass

    def _load(self, path):
        image = cv2.imread(path)

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        return image

    def _structure_histogram(self, image):
        """
        Create a brightness-resistant structural descriptor.

        Uses image gradients instead of raw color information.
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        gx = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            gx,
            gy
        )

        angle = cv2.phase(
            gx,
            gy,
            angleInDegrees=True
        )

        histogram = np.zeros(
            36,
            dtype=np.float32
        )

        bin_indices = (
            angle / 10.0
        ).astype(np.int32)

        bin_indices = np.clip(
            bin_indices,
            0,
            35
        )

        for i in range(36):
            histogram[i] = np.sum(
                magnitude[
                    bin_indices == i
                ]
            )

        histogram = histogram.reshape(-1, 1)

        cv2.normalize(
            histogram,
            histogram
        )

        return histogram

    def _similarity(
        self,
        reference,
        generated
    ):
        """
        Compare two structural descriptors.
        """

        score = cv2.compareHist(
            reference,
            generated,
            cv2.HISTCMP_CORREL
        )

        return max(
            0.0,
            min(
                1.0,
                float(score)
            )
        )

    def _spatial_structure_similarity(
        self,
        reference,
        generated,
        grid_size=3
    ):
        """
        Compare structural descriptors spatially.

        Instead of treating the whole object region as one
        histogram, divide it into a grid and compare each cell.
        This makes large local replacements easier to detect.
        """

        height, width = reference.shape[:2]

        scores = []

        for row in range(grid_size):

            for col in range(grid_size):

                y1 = int(
                    row * height / grid_size
                )

                y2 = int(
                    (row + 1) * height / grid_size
                )

                x1 = int(
                    col * width / grid_size
                )

                x2 = int(
                    (col + 1) * width / grid_size
                )

                ref_cell = reference[
                    y1:y2,
                    x1:x2
                ]

                gen_cell = generated[
                    y1:y2,
                    x1:x2
                ]

                ref_descriptor = (
                    self._structure_histogram(
                        ref_cell
                    )
                )

                gen_descriptor = (
                    self._structure_histogram(
                        gen_cell
                    )
                )

                score = self._similarity(
                    ref_descriptor,
                    gen_descriptor
                )

                scores.append(score)

        return float(np.mean(scores))

    def _edge_difference(
        self,
        reference,
        generated
    ):
        """
        Measure local edge-map difference.

        This complements the orientation histogram because
        large replacements can remove or introduce strong edges.
        """

        ref_gray = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY
        )

        gen_gray = cv2.cvtColor(
            generated,
            cv2.COLOR_BGR2GRAY
        )

        ref_edges = cv2.Canny(
            ref_gray,
            50,
            150
        )

        gen_edges = cv2.Canny(
            gen_gray,
            50,
            150
        )

        ref_edges = ref_edges.astype(
            np.float32
        ) / 255.0

        gen_edges = gen_edges.astype(
            np.float32
        ) / 255.0

        difference = np.mean(
            np.abs(
                ref_edges - gen_edges
            )
        )

        similarity = 1.0 - float(
            difference
        )

        return max(
            0.0,
            min(
                1.0,
                similarity
            )
        )

    def _object_score(
        self,
        reference,
        generated
    ):
        """
        Combine spatial structural similarity and
        edge similarity for the object region.
        """

        spatial_score = (
            self._spatial_structure_similarity(
                reference,
                generated,
                grid_size=3
            )
        )

        edge_score = (
            self._edge_difference(
                reference,
                generated
            )
        )

        # Spatial structure gets more weight because
        # object layout is the primary signal.
        score = (
            0.70 * spatial_score
            + 0.30 * edge_score
        )

        return round(
            float(score),
            4
        )

    def compare(
        self,
        reference_path,
        generated_path
    ):
        """
        Compare object and environment consistency
        between reference and generated images.
        """

        reference = self._load(
            reference_path
        )

        generated = self._load(
            generated_path
        )

        generated = cv2.resize(
            generated,
            (
                reference.shape[1],
                reference.shape[0]
            )
        )

        height, width = reference.shape[:2]

        # --------------------------------------------------
        # Object region
        # --------------------------------------------------

        y1 = int(height * 0.20)
        y2 = int(height * 0.80)

        x1 = int(width * 0.20)
        x2 = int(width * 0.80)

        ref_object = reference[
            y1:y2,
            x1:x2
        ]

        gen_object = generated[
            y1:y2,
            x1:x2
        ]

        # --------------------------------------------------
        # Environment region
        # --------------------------------------------------

        ref_environment = reference.copy()
        gen_environment = generated.copy()

        ref_environment[
            y1:y2,
            x1:x2
        ] = 0

        gen_environment[
            y1:y2,
            x1:x2
        ] = 0

        # --------------------------------------------------
        # Object comparison
        # --------------------------------------------------

        object_score = self._object_score(
            ref_object,
            gen_object
        )

        # --------------------------------------------------
        # Environment comparison
        # --------------------------------------------------

        environment_reference_descriptor = (
            self._structure_histogram(
                ref_environment
            )
        )

        environment_generated_descriptor = (
            self._structure_histogram(
                gen_environment
            )
        )

        environment_score = self._similarity(
            environment_reference_descriptor,
            environment_generated_descriptor
        )

        environment_score = round(
            environment_score,
            4
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        if object_score >= 0.70:
            object_label = "OBJECT_CONSISTENT"
        else:
            object_label = "OBJECT_DRIFT"

        if environment_score >= 0.70:
            environment_label = "ENVIRONMENT_CONSISTENT"
        else:
            environment_label = "ENVIRONMENT_DRIFT"

        return {
            "object": {
                "score": object_score,
                "label": object_label
            },
            "environment": {
                "score": environment_score,
                "label": environment_label
            }
        }


if __name__ == "__main__":

    checker = ObjectEnvironmentChecker()

    result = checker.compare(
        "mock_data/inputs/adaface/reference/reference_adaface.jpeg",
        "mock_data/inputs/adaface/generated/generated_brightness.jpeg"
    )

    print(result)