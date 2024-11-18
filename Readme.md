# Performance recorder

Simple tool to record CPU, memory, swap and possibly gpu usage.
Simply create a `.venv` with
```shell
python -m venv .venv
```
Then source it:
```shell
source .venv/bin/activate
```
And install the `requirements.txt`
```shell
pip install -r requirements.txt
```

Then we can execute the program:
```shell
python main.py --recording-time=10 --interval-seconds=1
```
To record data for 5 seconds, every 1 second without a GPU.
To enable GPU loggin, call:
```shell
python main.py --recording-time=10 --interval-seconds=1 --record-nvda-gpu=True

```