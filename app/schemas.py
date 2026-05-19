from pydantic import BaseModel


class HeartBeatResponse(BaseModel):
    is_alive: bool = True
