from utils import *
model_name = "deepset/roberta-large-squad2"
q = "How did hacks and theft affected Bitcoin"
limit_predict, limit_real_predict, limit_scores = test(q, model_name)
print("limit_predict", limit_predict)
print("limit_real_predict", limit_real_predict)
print("limit_scores", limit_scores)  

# limit_predict ['Together with slaves freed by their masters after the Revolutionary War and escaped slaves, a significant free-black population gradually developed in Manhattan.', 'The city was a haven for Loyalist refugees, as well as escaped slaves who joined the British lines for freedom newly promised by the 
# Crown for all fighters.', 'New York City was a prime destination in the early twentieth century for African Americans during the Great Migration from the American South, and by 1916, New York City was home to the largest urban African diaspora in North America.']
# limit_real_predict ['a significant free-black population gradually developed in Manhattan', 'joined the British lines', 'New York City was home to the largest urban African diaspora']
# limit_scores [0.22538922727108002, 0.07307355850934982, 0.003589204978197813]