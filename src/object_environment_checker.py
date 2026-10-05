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

        Uses image gradients instead of raw color information,
        making the comparison less sensitive to brightness changes.
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Normalize local contrast
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        # Calculate horizontal and vertical gradients
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

        # Gradient magnitude
        magnitude = cv2.magnitude(
            gx,
            gy
        )

        # Gradient orientation
        angle = cv2.phase(
            gx,
            gy,
            angleInDegrees=True
        )

        # --------------------------------------------------
        # Build orientation histogram manually
        # --------------------------------------------------
        #
        # We do this manually because the installed OpenCV
        # version does not support the "weights" argument
        # in cv2.calcHist().
        # --------------------------------------------------

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

        histogram = histogram.reshape(
            -1,
            1
        )

        # Normalize descriptor
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

        return round(
            max(
                0.0,
                min(
                    1.0,
                    float(score)
                )
            ),
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

        # --------------------------------------------------
        # Load images
        # --------------------------------------------------

        reference = self._load(
            reference_path
        )

        generated = self._load(
            generated_path
        )

        # Match generated image size to reference
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
        #
        # Approximate central object region.
        # This is a prototype region-based checker.
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
        # Object structural similarity
        # --------------------------------------------------

        object_reference_descriptor = (
            self._structure_histogram(
                ref_object
            )
        )

        object_generated_descriptor = (
            self._structure_histogram(
                gen_object
            )
        )

        object_score = self._similarity(
            object_reference_descriptor,
            object_generated_descriptor
        )

        # --------------------------------------------------
        # Environment structural similarity
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

        # --------------------------------------------------
        # Final result
        # --------------------------------------------------

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