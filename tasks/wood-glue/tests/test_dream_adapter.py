"""Adapter contract checks only: these make no inference request and claim no Dream score."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspect_robots import Observation,Scene
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import Chunk,DreamSession
from wood_glue.dream_run import AuditedDreamSession


def test_repository_policy_passes_model_rows_and_three_cameras_without_scripted_actions():
    class Session:
        calls=0
        observations=[]
        def start(self,episode):assert not episode.record
        def observe(self,frames,state,prompt,timestamp):
            assert set(frames)=={'left_cam','top_cam','right_cam'} and len(state)==14
            self.observations.append(timestamp)
        def act(self,commanded):
            self.calls+=1
            return Chunk(rows=np.tile(np.arange(14)*.01,(48,1)),rtt_s=.1,attempts=1,session_id='test-only')
        def close(self):pass
    session=Session();policy=DreamPolicy(session=session,preroll_steps=0,rows_per_request=2)
    policy.reset(Scene(id='test',instruction='Hold the glued corner.'))
    observation=Observation(images={n:np.zeros((176,320,3),np.uint8) for n in ('left_cam','top_cam','right_cam')},state={'joint_pos':np.zeros(14)},instruction='Hold the glued corner.')
    for _ in range(3):assert np.allclose(policy.act(observation).actions[0].data,np.arange(14)*.01)
    assert session.calls==2 and session.observations==[0.,.05,.1]
    policy.close()


def test_audit_rejects_a_restarted_model_session(monkeypatch,tmp_path):
    chunks=iter([Chunk(np.zeros((48,14)),.1,1,session_id='one'),Chunk(np.zeros((48,14)),.1,1,session_id='two')])
    monkeypatch.setattr(DreamSession,'act',lambda self,commanded:next(chunks))
    session=AuditedDreamSession(tmp_path)
    session.act(None)
    with pytest.raises(RuntimeError,match='session changed'):session.act(None)
    session.close()
