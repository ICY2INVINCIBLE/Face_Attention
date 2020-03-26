# DataSet
    · val_1429:There are 1429 validation photos
    · train_5358:There 5358 train photos
    · test_1630:There 1630 test photos
    · the name of photo like 3_150, the first one 3 is engagement, the second one 150 is the number which you can ignore. 

These photos are extracted one frame from each video and their resolution has been processed as 90*120. At the sametime, I have already extracted 10 frames from each video. Because gitbash only can upload the file small than 100M unless using git lfs and my Intrnet speeds are incredibly slow, therefore I give up downloading the gitlfs to upload these dataset. If you want to get, you can contact with me. <br>
**There are several incorrect photos haven't been processed.**<br>

# _CNN.py
**1.Tensorflow should lower than 2.**  <br>
**2.Line 198 and 199 change the path where your train and validation put.** <br>
**3.Line 153 change to your own path.**<br>

# Result
I used _CNN.py again and here are the best results.<br>
train loss: 0.872401<br>
train acc: 0.477598<br>
validation loss: 1.048778<br>
validation acc: 0.594460<br>
