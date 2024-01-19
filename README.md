# MatchingService

- Service based AI
- The service designed to serve Indoor Localization App
- It has a couple of APIs, Enrich, Find and Train.
    - Enrich API, to enable the APP enrich database with images and metadata on locations, by uploading video that
      describes the location.
    - In Enrich API, there is an ability to label existing data, by uploading image to label specific location in
      database.
    - Finds API, is an API that uses ML trained model, to find the location of uploaded image.
    - Train API, is an API to train the ML model upon the data existing in database,

## Indoor Localization Backend

**To run app, run the command in terminal/cmd:**

`python app.py
`

## Api Calls Examples:

#### _Enrichment API_

Video:

    curl --location 'http://{host}:{port}/api/enrich/video' \
    --form 'video=@"20230719_094627.mp4"' \
    --form 'name="The Phoenicians "' \
    --form 'description="Archaeology"'

Label:

    curl --location 'http://{host}:{port}/api/enrich/label' \
    --form 'image=@"20230719_094627/image1.jpg"' \
    --form 'class="648f39d5e60ce40af4d51d1c"'

#### _Find API_

#### _Train Model API_

    curl --location 'http://{host}:{port}/api/train'

#### _HealthCheck API_

    curl --location 'http://{host}:{port}/api/healthcheck'

## Service Architecture

![Screenshot 2023-08-18 at 10.06.14.png](files%2Fdesign%2FScreenshot%202023-08-18%20at%2010.06.14.png)

### TODO:

- Build training data in this: get top 3 results, get percent of results above threshold (mini research needed to decide
  about it) and add column that will contain the location zone.
  example data frame should look
  like: [top1_clip, top1_resnet, top2_clip, top2_resnet, top3_clip, top3_resnet, percent, zone]
- Capture a video from museum, then split it to frames, these frames compare them to the representative images that
  image_selector chose, then check if frames that captured should be represented by or not.
- In thesis proposal, update the Research question and goals by deleting the repetitive text, and replace it by talking
  on challenges.
- Need also to add primer results, to talk about results we have till now.
