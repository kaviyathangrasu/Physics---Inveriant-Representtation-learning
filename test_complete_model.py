import torch

from complete_model import CompleteChangeDetectionModel


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CREATE MODEL
# ============================================================

model = CompleteChangeDetectionModel(
    pretrained=True
)

model = model.to(device)

model.eval()

print("\nComplete model created successfully.")


# ============================================================
# CREATE TEST INPUTS
# ============================================================

batch_size = 4

image_size = 256


image1 = torch.randn(
    batch_size,
    3,
    image_size,
    image_size
).to(device)


image2 = torch.randn(
    batch_size,
    3,
    image_size,
    image_size
).to(device)


print(
    "\nImage1 shape:",
    image1.shape
)

print(
    "Image2 shape:",
    image2.shape
)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    outputs = model(
        image1,
        image2
    )


# ============================================================
# PRINT OUTPUTS
# ============================================================

print("\nComplete model outputs:\n")


for name, value in outputs.items():

    if isinstance(value, dict):

        print(
            f"{name}:"
        )

        for sub_name, sub_value in value.items():

            print(
                f"  {sub_name:25s}: "
                f"{sub_value.shape}"
            )

    else:

        print(
            f"{name:35s}: "
            f"{value.shape}"
        )


# ============================================================
# EXPECTED SHAPES
# ============================================================

expected_change = (
    4,
    1,
    256,
    256
)

expected_earth = (
    4,
    256,
    8,
    8
)

expected_feature1 = (
    4,
    256,
    64,
    64
)

expected_feature2 = (
    4,
    512,
    32,
    32
)

expected_feature3 = (
    4,
    1024,
    16,
    16
)

expected_feature4 = (
    4,
    2048,
    8,
    8
)


# ============================================================
# TEST STATUS
# ============================================================

tests_passed = True


# ============================================================
# CHANGE OUTPUT
# ============================================================

if outputs[
    "change_logits"
].shape != expected_change:

    print(
        "\nERROR: change_logits shape incorrect."
    )

    tests_passed = False


# ============================================================
# EARTH REPRESENTATIONS
# ============================================================

if outputs[
    "earth1"
].shape != expected_earth:

    print(
        "\nERROR: earth1 shape incorrect."
    )

    tests_passed = False


if outputs[
    "earth2"
].shape != expected_earth:

    print(
        "\nERROR: earth2 shape incorrect."
    )

    tests_passed = False


# ============================================================
# NUISANCE REPRESENTATIONS
# ============================================================

if outputs[
    "nuisance1"
].shape != expected_earth:

    print(
        "\nERROR: nuisance1 shape incorrect."
    )

    tests_passed = False


if outputs[
    "nuisance2"
].shape != expected_earth:

    print(
        "\nERROR: nuisance2 shape incorrect."
    )

    tests_passed = False


# ============================================================
# EARTH DIFFERENCE
# ============================================================

if outputs[
    "earth_difference"
].shape != expected_earth:

    print(
        "\nERROR: earth_difference shape incorrect."
    )

    tests_passed = False


# ============================================================
# ATTENTION FUSION
# ============================================================

if outputs[
    "fused_feature"
].shape != expected_earth:

    print(
        "\nERROR: fused_feature shape incorrect."
    )

    tests_passed = False


if outputs[
    "attention_map"
].shape != expected_earth:

    print(
        "\nERROR: attention_map shape incorrect."
    )

    tests_passed = False


# ============================================================
# TEMPORAL SKIP FEATURES
# ============================================================

if outputs[
    "temporal_feature1_difference"
].shape != expected_feature1:

    print(
        "\nERROR: temporal feature1 difference "
        "shape incorrect."
    )

    tests_passed = False


if outputs[
    "temporal_feature2_difference"
].shape != expected_feature2:

    print(
        "\nERROR: temporal feature2 difference "
        "shape incorrect."
    )

    tests_passed = False


if outputs[
    "temporal_feature3_difference"
].shape != expected_feature3:

    print(
        "\nERROR: temporal feature3 difference "
        "shape incorrect."
    )

    tests_passed = False


# ============================================================
# ENCODER FEATURES
# ============================================================

features1 = outputs[
    "encoder_features1"
]

features2 = outputs[
    "encoder_features2"
]


# ============================================================
# IMAGE 1 FEATURES
# ============================================================

if features1[
    "feature1"
].shape != expected_feature1:

    print(
        "\nERROR: Image1 feature1 shape incorrect."
    )

    tests_passed = False


if features1[
    "feature2"
].shape != expected_feature2:

    print(
        "\nERROR: Image1 feature2 shape incorrect."
    )

    tests_passed = False


if features1[
    "feature3"
].shape != expected_feature3:

    print(
        "\nERROR: Image1 feature3 shape incorrect."
    )

    tests_passed = False


if features1[
    "feature4"
].shape != expected_feature4:

    print(
        "\nERROR: Image1 feature4 shape incorrect."
    )

    tests_passed = False


# ============================================================
# IMAGE 2 FEATURES
# ============================================================

if features2[
    "feature1"
].shape != expected_feature1:

    print(
        "\nERROR: Image2 feature1 shape incorrect."
    )

    tests_passed = False


if features2[
    "feature2"
].shape != expected_feature2:

    print(
        "\nERROR: Image2 feature2 shape incorrect."
    )

    tests_passed = False


if features2[
    "feature3"
].shape != expected_feature3:

    print(
        "\nERROR: Image2 feature3 shape incorrect."
    )

    tests_passed = False


if features2[
    "feature4"
].shape != expected_feature4:

    print(
        "\nERROR: Image2 feature4 shape incorrect."
    )

    tests_passed = False


# ============================================================
# FINITE VALUE CHECK
# ============================================================

for name, value in outputs.items():

    if isinstance(value, dict):

        for sub_name, sub_value in value.items():

            if not torch.isfinite(
                sub_value
            ).all():

                print(
                    f"\nERROR: Non-finite values in "
                    f"{name} -> {sub_name}"
                )

                tests_passed = False

    else:

        if not torch.isfinite(
            value
        ).all():

            print(
                f"\nERROR: Non-finite values in "
                f"{name}"
            )

            tests_passed = False


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n======================================"
)

if tests_passed:

    print(
        "     TEMPORAL DECODER TEST PASSED"
    )

else:

    print(
        "     TEMPORAL DECODER TEST FAILED"
    )

print(
    "======================================"
)