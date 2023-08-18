# MatchingService

- Service based AI
- The service designed to serve Indoor Localization App
- It has a couple of APIs, Enrich, Find and Train.
    - Enrich API, to enable the APP enrich database with images and metadata on locations, by uploading video that
      describes the location.
    - In Enrich API, there is an ability to label existing data, by uploading image to label specific location in
      database.
    - Fins API, is an API that uses ML trained model, to find the location of uploaded image.
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

![Screenshot 2023-08-18 at 10.06.14.png](uploads%2Fdesign%2FScreenshot%202023-08-18%20at%2010.06.14.png)