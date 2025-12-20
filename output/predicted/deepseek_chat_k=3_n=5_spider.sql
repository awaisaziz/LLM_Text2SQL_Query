SELECT count(*) FROM singer
SELECT count(*) FROM singer
SELECT Name ,  Country ,  Age FROM singer ORDER BY Age DESC
SELECT Name ,  Country ,  Age FROM singer ORDER BY Age DESC
SELECT AVG(age) ,  MIN(age) ,  MAX(age) FROM singer WHERE Country  =  'France'
SELECT AVG(age) ,  MIN(age) ,  MAX(age) FROM singer WHERE Country  =  'French'
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Age  =  (SELECT min(Age) FROM singer)
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Age = (SELECT min(Age) FROM singer)
SELECT DISTINCT Country FROM singer WHERE Age > 20
SELECT DISTINCT Country FROM singer WHERE Age > 20
SELECT Country ,  COUNT(*) FROM singer GROUP BY Country
SELECT COUNT(*) ,  T1.Country FROM singer AS T1 GROUP BY T1.Country
SELECT T1.Song_Name FROM singer AS T1 WHERE T1.Age > (SELECT avg(Age) FROM singer)
SELECT T1.Song_Name FROM singer AS T1 WHERE T1.Age > (SELECT avg(Age) FROM singer)
SELECT Location ,  Name FROM stadium WHERE Capacity BETWEEN 5000 AND 10000
SELECT Location ,  Name FROM stadium WHERE Capacity BETWEEN 5000 AND 10000
SELECT max(Capacity) ,  avg(Capacity) FROM stadium
SELECT avg(Capacity) ,  max(Capacity) FROM stadium
SELECT Name ,  Capacity FROM stadium ORDER BY Average DESC LIMIT 1
SELECT Name ,  Capacity FROM stadium ORDER BY Average DESC LIMIT 1
SELECT count(*) FROM concert WHERE Year = 2014 OR Year = 2015
SELECT count(*) FROM concert WHERE Year = 2014 OR Year = 2015
SELECT T1.Name ,  COUNT(*) FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID GROUP BY T1.Stadium_ID
SELECT T1.Stadium_ID ,  T1.Location ,  T1.Name ,  count(T2.concert_ID) FROM stadium AS T1 LEFT JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID GROUP BY T1.Stadium_ID ,  T1.Location ,  T1.Name
SELECT Name ,  Capacity FROM stadium WHERE Stadium_ID IN (SELECT Stadium_ID FROM concert WHERE Year >= 2014 GROUP BY Stadium_ID ORDER BY COUNT(*) DESC LIMIT 1)
SELECT Name ,  Capacity FROM stadium WHERE Stadium_ID IN (SELECT Stadium_ID FROM concert WHERE Year > 2013 GROUP BY Stadium_ID ORDER BY count(*) DESC LIMIT 1)
SELECT Year FROM concert GROUP BY Year ORDER BY COUNT(*) DESC LIMIT 1
SELECT Year FROM concert GROUP BY Year ORDER BY COUNT(*) DESC LIMIT 1
SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert)
SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert)
SELECT Country FROM singer WHERE Age  >  40 INTERSECT SELECT Country FROM singer WHERE Age  <  30
SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert WHERE Year = 2014)
SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert WHERE Year = 2014)
SELECT T1.concert_Name ,  T1.Theme ,  count(T2.Singer_ID) FROM concert AS T1 JOIN singer_in_concert AS T2 ON T1.concert_ID  =  T2.concert_ID GROUP BY T1.concert_ID
SELECT T1.concert_Name ,  T1.Theme ,  count(*) FROM concert AS T1 JOIN singer_in_concert AS T2 ON T1.concert_ID  =  T2.concert_ID GROUP BY T1.concert_ID
SELECT T1.Name ,  count(*) FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID  =  T2.Singer_ID GROUP BY T1.Singer_ID
SELECT T1.Name ,  count(*) FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID  =  T2.Singer_ID GROUP BY T1.Singer_ID
SELECT DISTINCT T2.Name FROM concert AS T1 JOIN singer_in_concert AS T3 ON T1.concert_ID  =  T3.concert_ID JOIN singer AS T2 ON T3.Singer_ID  =  T2.Singer_ID WHERE T1.Year  =  2014
SELECT DISTINCT T1.Name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID  =  T2.Singer_ID JOIN concert AS T3 ON T2.concert_ID  =  T3.concert_ID WHERE T3.Year  =  2014
SELECT DISTINCT T1.Name ,  T1.Country FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID  =  T2.Singer_ID WHERE T1.Song_Name LIKE '%Hey%'
SELECT DISTINCT T1.Name ,  T1.Country FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID  =  T2.Singer_ID WHERE T1.Song_Name LIKE '%Hey%'
SELECT DISTINCT T1.Name ,  T1.Location FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Year  =  2014 INTERSECT SELECT DISTINCT T1.Name ,  T1.Location FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Year  =  2015
SELECT DISTINCT T1.Name ,  T1.Location FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Year  =  2014 INTERSECT SELECT DISTINCT T1.Name ,  T1.Location FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Year  =  2015
SELECT count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Capacity  =  (SELECT max(Capacity) FROM stadium)
SELECT count(*) FROM concert WHERE Stadium_ID = (SELECT Stadium_ID FROM stadium ORDER BY Capacity DESC LIMIT 1)
SELECT count(*) FROM Pets WHERE weight  >  10
SELECT count(*) FROM Pets WHERE weight > 10
SELECT T1.weight FROM Pets AS T1 WHERE T1.PetType  =  'dog' AND T1.pet_age  =  ( SELECT min(T2.pet_age) FROM Pets AS T2 WHERE T2.PetType  =  'dog' )
SELECT min(weight) FROM Pets WHERE PetType  =  'dog'
SELECT PetType ,  max(weight) FROM Pets GROUP BY PetType
SELECT PetType ,  max(weight) FROM Pets GROUP BY PetType
SELECT count(*) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID WHERE T1.Age  >  20
SELECT count(*) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID WHERE T1.Age  >  20
SELECT COUNT(*) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T1.Sex = 'F' AND T3.PetType = 'dog'
SELECT count(*) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T1.Sex  =  'F' AND T3.PetType  =  'dog'
SELECT count(DISTINCT PetType) FROM Pets
SELECT count(DISTINCT PetType) FROM Pets
SELECT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  'cat' OR T3.PetType  =  'dog'
SELECT DISTINCT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T3.PetType = "cat" OR T3.PetType = "dog"
SELECT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  'cat' INTERSECT SELECT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  'dog'
SELECT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  "cat" INTERSECT SELECT T4.Fname FROM Student AS T4 JOIN Has_Pet AS T5 ON T4.StuID  =  T5.StuID JOIN Pets AS T6 ON T5.PetID  =  T6.PetID WHERE T6.PetType  =  "dog"
SELECT Major ,  Age FROM Student WHERE StuID NOT IN (SELECT T1.StuID FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  'cat')
SELECT T1.Major ,  T1.Age FROM Student AS T1 WHERE T1.StuID NOT IN (SELECT T2.StuID FROM Has_Pet AS T2 JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  "cat")
SELECT StuID FROM Student EXCEPT SELECT T1.StuID FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  "cat"
SELECT StuID FROM Student EXCEPT SELECT T1.StuID FROM Has_Pet AS T1 JOIN Pets AS T2 ON T1.PetID = T2.PetID WHERE T2.PetType = "cat"
SELECT T1.Fname ,  T1.Age FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  'dog' AND T1.StuID NOT IN (SELECT T4.StuID FROM Has_Pet AS T4 JOIN Pets AS T5 ON T4.PetID  =  T5.PetID WHERE T5.PetType  =  'cat')
SELECT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T3.PetType = 'dog' AND T1.StuID NOT IN (SELECT T4.StuID FROM Has_Pet AS T4 JOIN Pets AS T5 ON T4.PetID = T5.PetID WHERE T5.PetType = 'cat')
SELECT PetType ,  weight FROM Pets WHERE pet_age  =  (SELECT min(pet_age) FROM Pets)
SELECT T1.PetType ,  T1.weight FROM Pets AS T1 ORDER BY T1.pet_age ASC LIMIT 1
SELECT PetID ,  weight FROM Pets WHERE pet_age  >  1
SELECT PetID ,  weight FROM Pets WHERE pet_age  >  1
SELECT avg(pet_age) ,  max(pet_age) FROM Pets GROUP BY PetType
SELECT avg(pet_age) ,  max(pet_age) ,  PetType FROM Pets GROUP BY PetType
SELECT PetType ,  avg(weight) FROM Pets GROUP BY PetType
SELECT PetType ,  avg(weight) FROM Pets GROUP BY PetType
SELECT DISTINCT T1.Fname ,  T1.Age FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID
SELECT DISTINCT T1.Fname ,  T1.Age FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID
SELECT T2.PetID FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID WHERE T1.Lname  =  'Smith'
SELECT T2.PetID FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID WHERE T1.LName  =  'Smith'
SELECT T1.StuID ,  count(*) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID GROUP BY T1.StuID
SELECT T1.StuID ,  count(*) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID GROUP BY T1.StuID
SELECT T1.Fname ,  T1.Sex FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID GROUP BY T1.StuID HAVING COUNT(*)  >  1
SELECT T1.Fname ,  T1.Sex FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID GROUP BY T1.StuID HAVING COUNT(*)  >  1
SELECT T1.LName FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  "cat" AND T3.pet_age  =  3
SELECT T1.Lname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T3.PetType = "cat" AND T3.pet_age = 3
SELECT avg(T1.Age) FROM Student AS T1 WHERE T1.StuID NOT IN ( SELECT StuID FROM Has_Pet )
SELECT avg(T1.Age) FROM Student AS T1 WHERE T1.StuID NOT IN (SELECT T2.StuID FROM Has_Pet AS T2)
SELECT COUNT(DISTINCT Continent) FROM continents
SELECT COUNT(DISTINCT Continent) FROM continents
SELECT T1.ContId ,  T1.Continent ,  count(*) FROM continents AS T1 JOIN countries AS T2 ON T1.Continent  =  T2.Continent GROUP BY T1.ContId ,  T1.Continent
SELECT T1.ContId ,  T1.Continent ,  COUNT(*) FROM continents AS T1 JOIN countries AS T2 ON T1.ContId  =  T2.Continent GROUP BY T1.ContId ,  T1.Continent
SELECT count(DISTINCT CountryName) FROM countries
SELECT COUNT(DISTINCT CountryName) FROM countries
SELECT T1.FullName ,  T1.Id ,  COUNT(*) FROM car_makers AS T1 JOIN model_list AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.FullName ,  T1.Id
SELECT T1.Id ,  T1.FullName ,  count(T2.ModelId) FROM car_makers AS T1 JOIN model_list AS T2 ON T1.Id  =  T2.Maker GROUP BY T2.Maker
SELECT Model FROM model_list WHERE Model IN (SELECT Model FROM car_names WHERE MakeId IN (SELECT Id FROM cars_data WHERE Horsepower = (SELECT MIN(Horsepower) FROM cars_data)))
SELECT T1.Model FROM model_list AS T1 JOIN cars_data AS T2 ON T1.ModelId  =  T2.Id ORDER BY T2.Horsepower ASC LIMIT 1
SELECT Model FROM model_list WHERE Model IN (SELECT T1.Model FROM car_names AS T1 JOIN cars_data AS T2 ON T1.MakeId = T2.Id WHERE T2.Weight < (SELECT avg(Weight) FROM cars_data))
SELECT Model FROM model_list WHERE Model IN (SELECT Model FROM car_names WHERE MakeId IN (SELECT Id FROM cars_data WHERE Weight < (SELECT avg(Weight) FROM cars_data)))
SELECT DISTINCT T1.Maker FROM car_makers AS T1 JOIN model_list AS T2 ON T1.Id = T2.Maker JOIN car_names AS T3 ON T2.Model = T3.Model JOIN cars_data AS T4 ON T3.MakeId = T4.Id WHERE T4.Year = 1970
