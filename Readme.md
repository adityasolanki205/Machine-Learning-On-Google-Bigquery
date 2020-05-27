# Supervised Learning Using Bigquery

This is a Machine Learning model using GCP's [Bigquery](https://cloud.google.com/bigquery-ml/docs/bigqueryml-intro).Here I have tried to perform Supervised Machine learning on [Titanic dataset](https://www.kaggle.com/c/titanic) from [Kaggle](https://www.kaggle.com/). Supervised Learning on GCP could be divided into five simple steps:

1. **Data Exploration**
2. **Data Wrangling**
3. **Model Creation**
4. **Model Evaluation**
5. **Model Implementation**


## Motivation
For the last one year, I have been part of a great learning curve wherein I have upskilled myself to move into a Machine Learning and Cloud Computing. This project was practice project for all the learnings I have had. This is first of the many more to come. 
 

## Database used

<b>Built with</b>
- [Bigquery](https://cloud.google.com/bigquery-ml/docs/bigqueryml-intro)


## Code Example

```bash
    # clone this repo, removing the '-' to allow python imports:
    git clone https://github.com/adityasolanki205/Machine-Learning-On-Google-Bigquery.git
```

## Installation

Below are the steps to setup the enviroment and run the codes:

1. **Data Exploration**: First the data exploration has to be done. Download the dataset from [Titanic dataset](https://www.kaggle.com/c/titanic). Steps to upload the data and its exploration are given below.
    
    a. Create a Google Cloud Bucket.
    
    b. Upload the Train and test CSV files in that Bucket
    
    c. Goto the Bisquery using the Navigation Menu on the Top Left
    
    d. Create a Dataset 
    
    e. Create a table using Google Cloud Storage and auto detecting the Schema
    
    f. Select the correct Csv file
    
    g. Use Simple SQL queries, try to find important points and relevant data in the dataset. 
       It will be used while data wrangling. For eg. 
       
```sql
    # Here we are selecting Survived Passengers grouped by their sex
    Select 
        Sex , 
        Survived,
        count(*)
    FROM `daring-span-249015.titanic_dataset.Train`
    Group by  
        Survived , 
        Sex
    order by 
        1 , 
        2;
```
![](Images/Sex_Survived.PNG)

2. **Data Wrangling**: After exploration we will try to clean, structure and enrich the data so that it can be make more sense to the Algorithm.


```sql
    # Here We are trying to select passengers according to their age group. 
    select 
        case
            when age <18 then 'Under 18'
            when age between 18 and 24 then '18-24'
            when age between 25 and 34 then '25-34'
            when age > 34 then '35 above'
        END as age_range, 
        Count(*) as count
    from `daring-span-249015.titanic_dataset.Train`
    group by age_range
    order by age_range;
```
![](Images/Age.PNG)

```sql
    # So we will update the null Values with Mean of the complete column
    Update `daring-span-249015.titanic_dataset.Train`
    set age = (SELECT avg(age)
                from `daring-span-249015.titanic_dataset.Train`
                where age IS NOT NULL
                )
    where age IS NULL;
```
![](Images/age_updated.PNG)

3. **Model Creation**: After face extraction we will fetch the face embedding using [FaceNet](https://github.com/davidsandberg/facenet). Downloaded the model [here](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn). After running this code for all the faces in train and test folders, we can save the embeddings using [np.saves_compressed](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)

```python
    # The Dimension of the input has to be increased as the model expects input in the form (Sample size, 160, 160,3)
    samples = np.expand_dims(image_pixels, axis = 0)
    
    # Use the Predict method to find the Embeddings in the face. Output would be 1D vector of 128 embeddings of that face
    embeddings = model.predict(samples)
```

4. **Model Evaluation**:  Now we will train SVM model over the embeddings to predict the face of a person.

```python
    # We will use Linear SVM model to train over the embeddings
    model = SVC(kernel = 'linear', probability=True).fit(X_train,y_train)
```

5. **Model Implementation**: After the training of SVM model we will predict the face over test dataset.

```python
    # Preprocessing of the test photos have to be done like we did for Train and Validation photos
    image = np.asarray(image.convert('RGB'))
    
    # Now extract the face
    faces = MTCNN.detect_faces(image)
    
    # Extract embeddings
    embeddings = model.predict(samples)
    
    # At last we will predict the face embeddings
    SVM_model.predict(X_test)
```

## Tests
To test the code we need to do the following:

    1. Copy the photo to be tested in 'Test' subfolder of 'Data' folder. 
    Here I have used a photo of Elton John and Madonna
![](data/test/singers.jpg)
    
    2. Goto the 'Predict face in a group' folder.
    
    3. Open the 'Predict from a group of faces.ipynb'
    
    4. Goto filename variable and provide the path to your photo. Atlast run the complete code. 
    The recognised faces would have been highlighted and a photo would be saved by the name 'Highlighted.jpg'
![](output.jpg)

**Note**: The boundary boxes are color coded:

    1. Aditya Solanki  : Yellow
    2. Ben Afflek      : Blue   
    3. Elton John      : Green
    4. Jerry Seinfield : Red
    5. Madonna         : Aqua
    6. Mindy Kaling    : White
    
## How to use?
To run the complete code, follow the process below:

    1. Create Data Folder. 
    
    2. Create Sub folders as Training and Validation Dataset
    
    3. Create all the celebrity folders with all the required photos in them. 
    
    4. Run the Train and Test Data.ipynb file under Training Data Creation folder
    
    5. Save the output as numpy arrays
    
    6. Run the Face embedding using FaceNet.ipynb under the same folder name. This will create training data for SVM model
    
    7. Run the Predict from a group of faces.ipynb to recognise a familiar face

## Credits
1. David Sandberg's facenet repo: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)
2. Tim Esler's Git repo:[https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
3. Akash Nimare's README.md: https://gist.github.com/akashnimare/7b065c12d9750578de8e705fb4771d2f#file-readme-md
4. [Machine learning mastery](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)
