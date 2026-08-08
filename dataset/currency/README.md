# dataset/currency/

Add clear, front-facing reference photos of each currency note here,
named by denomination, e.g.:

```
dataset/currency/10.jpg
dataset/currency/20.jpg
dataset/currency/50.jpg
dataset/currency/100.jpg
dataset/currency/200.jpg
dataset/currency/500.jpg
```

`modules/currency_detector.py` uses ORB feature matching against these
reference images to guess which denomination is shown to the camera when
you press "Detect Currency" on the Live Camera page. This is a
lightweight heuristic, not a trained model - accuracy depends heavily on
lighting, note condition, and image quality.

Without any reference images here, currency detection simply reports
"not recognized" instead of crashing.

For a genuinely robust solution, replace this module with a trained
classifier (e.g. a small CNN or a YOLO classification head) on a proper
labeled currency-note dataset.
