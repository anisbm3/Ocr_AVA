from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions
import traceback

def dummy(audio):
    return b''

try:
    s = Stream(ReplyOnPause(dummy,
                             algo_options=AlgoOptions(audio_chunk_duration=1.5),
                             can_interrupt=False,
                             model_options=SileroVadOptions(threshold=0.5)
                             ), modality='audio', mode='send-receive')
    print('Stream created:', s)
except Exception as e:
    traceback.print_exc()
