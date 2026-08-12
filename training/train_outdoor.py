"""
Outdoor Detection Model Training - CURRENTLY NOT POSSIBLE
==============================================================
This script deliberately does NOT train a model. It exists to make
the outdoor domain's real blocker discoverable from training/ (where
every other domain has a working trainer) instead of silently having
no file at all.

Why outdoor cannot be trained right now (verified via
tools/analyze_datasets.py, not assumed):

    dataset/outdoor/XML Files/ contains 861 real Pascal VOC XML
    annotation files covering genuinely useful Indian-road classes -
    car (469), pole (160), truck (143), flyover (97), hoarding (43),
    traffic symbols (32), pedestrian (27), traffic signal (12),
    building (9), bus (9), bike (6), auto rickshaw (3), caravan (3) -
    1,013 real annotated objects after stripping the Roboflow
    watermark/junk labels ("day2 - v3 ...", "- collaborate with your
    team...", etc. - 2,899 of them).

    BUT every single one of those 861 XML files has NO corresponding
    source image anywhere in dataset/outdoor/ (or anywhere else in
    dataset/). The only actual image files present under outdoor/ are
    the unannotated Day2fog/ and Day2Rainy/ domain-transfer pairs
    (clear vs. foggy/rainy versions of the same scenes, no bounding
    boxes at all) and the OBB labels/ set, whose class-id mapping is
    undocumented and inconsistent across its Day/Foggy/Rainy subsets
    (Day uses ids 0-3, Rainy uses ids 0-13, no classes.txt anywhere)
    and so cannot be safely used either.

    tools/prepare_datasets.py --dataset outdoor already reflects this
    honestly: it converts every XML annotation it can, but produces
    dataset_prepared/outdoor/{train,val}/images with 0 images in it,
    because there is nothing to copy.

MISSING: source images for dataset/outdoor/XML Files/*.xml
RECOMMENDATION: capture (or otherwise obtain) real photos of Indian
road scenes and pair each one with one of the existing XML files -
the annotations describe real, valuable classes (car, pole, truck,
flyover, hoarding, traffic symbols, pedestrian, traffic signal, bus,
building, bike, auto rickshaw, caravan) and are ready to convert the
moment matching images exist. Alternatively, if the Day/Foggy/Rainy
labels/ OBB set's class-id mapping can be obtained from whoever
exported it, that set does have real images and could become a second
path to an outdoor model.

Until then, the outdoor domain is honestly reported as NOT AVAILABLE
by modules/model_router.py, and the app will say so rather than ever
silently substituting a generic COCO model for it.
"""


def main():
    print(__doc__)


if __name__ == "__main__":
    main()
