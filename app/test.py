from ultralytics import YOLO
from pathlib import Path
import cv2

# Load model (COCO classes, general objects)
model = YOLO("yolov8n.pt")

# Use a safe path
image_path = Path("app") / "Spicy-Penne-Pasta_-done.png"

# Run inference (set save=True to auto-save annotated image)
results = model(str(image_path))

res = results[0]            # first image result

# List detections
print("Detections:")
for box in res.boxes:
    cls_id = int(box.cls[0])
    name = res.names[cls_id]
    conf = float(box.conf[0])
    print(f" - {name}: {conf:.2f}")

# Save annotated image manually
annotated = res.plot()      # BGR image
cv2.imwrite("annotated.png", annotated)
print("Annotated image saved to annotated.png")

# (Optional) Display (only if you have GUI support)
# cv2.imshow("Result", annotated); cv2.waitKey(0); cv2.destroyAllWindows()
