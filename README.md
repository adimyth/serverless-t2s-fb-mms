## Deploying Huggingface Models on RunPod

### Running the project

1. Clone the repository
2. Install the requirements
```bash
pip install -r builder/requirements.txt
```
3. Run the project. Refer the [docs](https://docs.runpod.io/serverless/workers/development/overview) for more options.
```bash
python3 src/handler.py --rp_serve_api --rp_api_host 0.0.0.0 --rp_log_level DEBUG
```
4. Test the project
```bash
python src/handler.py --test_input '{"input": {"sentence": "प्रत्येक व्यक्ति को शिक्षा का अधिकार है । शिक्षा कम से कम प्रारम्भिक और बुनियादी अवस्थाओं में निःशुल्क होगी ।", "language": "hi"}}'
```