set -e
cd "$(dirname "$0")"
source .venv/bin/activate

LOGS="./output/logs"
mkdir -p "$LOGS"

echo "Launching all experiments in parallel..."

# M1 — AlexNet
python main.py --model_type alexnet --epochs 100 --lr 1e-4

# M2 — ResNet34
python main.py --model_type resnet34 --epochs 100 --lr 1e-4

# M3 — MobileNetV3
python main.py --model_type mobilenet_v3_large --epochs 100 --lr 1e-4


echo "All 9 experiments running. Waiting for completion..."
wait
echo "All done. Results in ./output/. Logs in $LOGS/"
