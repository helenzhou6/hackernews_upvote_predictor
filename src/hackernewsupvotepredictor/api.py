from fastapi import FastAPI
from pydantic import BaseModel
from hackernewsupvotepredictor.pred_model_for_api_data import get_predicted_score
app = FastAPI()

# uvicorn src.hackernewsupvotepredictor.api:app --reload to run and see it http://localhost:8000/healthcheck
@app.get("/healthcheck")
async def healthcheck():
      return "Up and running"

class PostData(BaseModel):
    title: str
    user_created: int
    time: int

# run send_post.sh to send data to see
@app.post("/how_many_upvotes")
def how_many_upvotes(post: PostData):
  pred_score = get_predicted_score(post)
  return {"pred_upvotes": pred_score}