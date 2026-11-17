"""The repository's installed Dream adapter with a local, continuous-session audit trail."""
import json
from inspect_robots_dream.session import DreamSession


class AuditedDreamSession(DreamSession):
    def __init__(self,out):
        super().__init__()
        self.replies=(out/'dream-replies.jsonl').open('x')
        self.previous_session=None
        self.observations=0

    def observe(self,frames,state,prompt,timestamp):
        self.observations+=1
        super().observe(frames,state,prompt,timestamp)

    def act(self,commanded):
        chunk=super().act(commanded)
        if self.previous_session is not None and chunk.session_id!=self.previous_session:
            raise RuntimeError('Dream session changed during the trial; this is not a continuous model rollout.')
        self.previous_session=chunk.session_id
        self.replies.write(json.dumps(dict(observation_step=self.observations,**chunk.to_json()))+'\n')
        self.replies.flush()
        return chunk

    def close(self):
        super().close()
        self.replies.close()
