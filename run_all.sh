set -e
cd "$(dirname "$0")"
source .venv/bin/activate

LOGS="./output/logs"
mkdir -p "$LOGS"

# M1 — AlexNet
python main.py --model_type alexnet --epochs 50

# M2 — ResNet34
python main.py --model_type resnet34 --epochs 50

# M3 — MobileNetV3
python main.py --model_type mobilenet_v3_large --epochs  50

wait
echo "All done. Results in ./output/. Logs in $LOGS/"
