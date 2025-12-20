SELECT count(*) FROM singer
SELECT count(*) FROM singer
SELECT Name, Country, Age FROM singer ORDER BY Age DESC
SELECT Name, Country, Age FROM singer ORDER BY Age DESC
SELECT avg(Age) ,  min(Age) ,  max(Age) FROM singer WHERE Country  =  'France'
SELECT AVG(Age), MIN(Age), MAX(Age) FROM singer WHERE Country = 'France'
SELECT Song_Name ,  Song_release_year FROM singer WHERE Age = (SELECT MIN(Age) FROM singer)
SELECT Song_Name ,  Song_release_year FROM singer WHERE Age  =  (SELECT MIN(Age) FROM singer)
SELECT DISTINCT Country FROM singer WHERE Age > 20
SELECT DISTINCT Country FROM singer WHERE Age > 20
SELECT Country ,  COUNT(*) FROM singer GROUP BY Country
SELECT Country ,  COUNT (*) FROM singer GROUP BY Country
SELECT Song_Name FROM singer WHERE Age > (SELECT AVG(Age) FROM singer)
SELECT Song_Name FROM singer WHERE Age > (SELECT avg(Age) FROM singer)
SELECT Location, Name FROM stadium WHERE Capacity BETWEEN 5000 AND 10000
SELECT Location, Name FROM stadium WHERE Capacity BETWEEN 5000 AND 10000
SELECT max(Capacity) ,  avg(Capacity) FROM stadium
SELECT avg(Capacity) ,  max(Capacity) FROM stadium
SELECT Name ,  Capacity FROM stadium ORDER BY Average DESC LIMIT 1
SELECT Name, Capacity FROM stadium ORDER BY Average DESC LIMIT 1
SELECT COUNT(*) FROM concert WHERE Year = 2014 OR Year = 2015
SELECT COUNT(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.Stadium_ID = T2.Stadium_ID WHERE T1.Year IN (2014, 2015)
SELECT T1.Name ,  count(*) FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID GROUP BY T1.Stadium_ID
SELECT T1.Name ,  COUNT(T2.concert_ID) FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID GROUP BY T1.Stadium_ID ,  T1.Name
SELECT T1.Name ,  T1.Capacity FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Year  >=  2014 GROUP BY T1.Stadium_ID ORDER BY count(*) DESC LIMIT 1
SELECT T1.Name ,  T1.Capacity FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Year > 2013 GROUP BY T1.Stadium_ID ORDER BY count(*) DESC LIMIT 1
SELECT Year FROM concert GROUP BY Year ORDER BY count(*) DESC LIMIT 1
SELECT Year FROM concert GROUP BY Year ORDER BY count(*) DESC LIMIT 1
SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert)
SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert)
SELECT Country FROM singer WHERE Age  >  40 INTERSECT SELECT Country FROM singer WHERE Age  <  30
SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert WHERE Year = 2014)
SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert WHERE Year = 2014)
SELECT T1.concert_Name ,  T1.Theme ,  count(T2.Singer_ID) FROM concert AS T1 LEFT JOIN singer_in_concert AS T2 ON T1.concert_ID  =  T2.concert_ID GROUP BY T1.concert_ID
SELECT T1.concert_Name ,  T1.Theme ,  count(T2.Singer_ID) FROM concert AS T1 JOIN singer_in_concert AS T2 ON T1.concert_ID  =  T2.concert_ID GROUP BY T1.concert_ID ,  T1.concert_Name ,  T1.Theme
SELECT T1.Name, count(*) FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID GROUP BY T1.Singer_ID
SELECT T1.Name ,  count(*) FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID  =  T2.Singer_ID GROUP BY T1.Singer_ID
SELECT DISTINCT T1.Name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID  =  T2.Singer_ID JOIN concert AS T3 ON T2.concert_ID  =  T3.concert_ID WHERE T3.Year  =  2014
SELECT DISTINCT T1.Name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID JOIN concert AS T3 ON T2.concert_ID = T3.concert_ID WHERE T3.Year = 2014
SELECT DISTINCT Name ,  Country FROM singer WHERE Song_Name LIKE '%Hey%'
SELECT T1.Name ,  T1.Country FROM singer AS T1 WHERE T1.Song_Name LIKE '%Hey%'
SELECT Name ,  Location FROM stadium WHERE Stadium_ID IN (SELECT Stadium_ID FROM concert WHERE Year = 2014 INTERSECT SELECT Stadium_ID FROM concert WHERE Year = 2015)
SELECT DISTINCT T1.Name ,  T1.Location FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Year  =  2014 INTERSECT SELECT DISTINCT T1.Name ,  T1.Location FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Year  =  2015
SELECT COUNT(*) FROM concert WHERE Stadium_ID = (SELECT Stadium_ID FROM stadium ORDER BY Capacity DESC LIMIT 1)
SELECT COUNT(*) FROM concert WHERE Stadium_ID = (SELECT Stadium_ID FROM stadium WHERE Capacity = (SELECT MAX(Capacity) FROM stadium))
SELECT COUNT(*) FROM Pets WHERE weight > 10
SELECT COUNT(*) FROM Pets WHERE weight > 10
SELECT weight FROM Pets WHERE PetType = 'dog' AND pet_age = (SELECT min(pet_age) FROM Pets WHERE PetType = 'dog')
SELECT min(weight) FROM Pets WHERE PetType  =  'dog' AND pet_age  =  (SELECT min(pet_age) FROM Pets WHERE PetType  =  'dog')
SELECT max(weight) ,  PetType FROM Pets GROUP BY PetType
SELECT PetType ,  MAX(weight) FROM Pets GROUP BY PetType
SELECT COUNT(*) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID WHERE T1.Age  >  20
SELECT COUNT(*) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID WHERE T1.Age  >  20
SELECT COUNT(*) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T1.Sex = "F" AND T3.PetType = "dog"
SELECT COUNT(DISTINCT T3.PetID) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T1.Sex = 'female' AND T3.PetType = 'dog'
SELECT count(DISTINCT PetType) FROM Pets
SELECT count(DISTINCT PetType) FROM Pets
SELECT DISTINCT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T3.PetType IN ('cat', 'dog')
SELECT Student.Fname FROM Student JOIN Has_Pet ON Student.StuID = Has_Pet.StuID JOIN Pets ON Has_Pet.PetID = Pets.PetID WHERE Pets.PetType = "cat" OR Pets.PetType = "dog"
SELECT DISTINCT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T3.PetType = 'cat' INTERSECT SELECT DISTINCT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T3.PetType = 'dog'
SELECT DISTINCT T1.Fname FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  'cat' AND T1.StuID IN (SELECT T4.StuID FROM Has_Pet AS T4 JOIN Pets AS T5 ON T4.PetID  =  T5.PetID WHERE T5.PetType  =  'dog')
SELECT Major, Age FROM Student WHERE StuID NOT IN (SELECT T1.StuID FROM Has_Pet AS T1 JOIN Pets AS T2 ON T1.PetID = T2.PetID WHERE T2.PetType = 'cat')
SELECT T1.Major ,  T1.Age FROM Student AS T1 WHERE T1.StuID NOT IN ( SELECT DISTINCT T2.StuID FROM Has_Pet AS T2 JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  "cat" )
SELECT StuID FROM Student EXCEPT SELECT StuID FROM Has_Pet JOIN Pets ON Has_Pet.PetID  =  Pets.PetID WHERE PetType  =  "cat"
SELECT StuID FROM Student EXCEPT SELECT StuID FROM Has_Pet JOIN Pets ON Has_Pet.PetID = Pets.PetID WHERE PetType = 'cat'
SELECT DISTINCT T1.Fname ,  T1.Age FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  "dog" AND T1.StuID NOT IN (SELECT T4.StuID FROM Has_Pet AS T4 JOIN Pets AS T5 ON T4.PetID  =  T5.PetID WHERE T5.PetType  =  "cat")
SELECT Fname FROM Student AS T1 WHERE EXISTS (SELECT 1 FROM Has_Pet AS T2 JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T2.StuID = T1.StuID AND T3.PetType = 'dog') AND NOT EXISTS (SELECT 1 FROM Has_Pet AS T4 JOIN Pets AS T5 ON T4.PetID = T5.PetID WHERE T4.StuID = T1.StuID AND T5.PetType = 'cat')
SELECT PetType ,  weight FROM Pets ORDER BY pet_age ASC LIMIT 1
SELECT PetType ,  weight FROM Pets WHERE pet_age = (SELECT min(pet_age) FROM Pets)
SELECT PetID ,  weight FROM Pets WHERE pet_age  >  1
SELECT PetID ,  weight FROM Pets WHERE pet_age  >  1
SELECT avg(pet_age) ,  max(pet_age) ,  PetType FROM Pets GROUP BY PetType
SELECT PetType, AVG(pet_age), MAX(pet_age) FROM Pets GROUP BY PetType
SELECT PetType ,  avg(weight) FROM Pets GROUP BY PetType
SELECT PetType ,  avg(weight) FROM Pets GROUP BY PetType
SELECT DISTINCT T1.Fname ,  T1.Age FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID
SELECT DISTINCT T1.Fname ,  T1.Age FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID
SELECT T2.PetID FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID WHERE T1.LName = 'Smith'
SELECT T2.PetID FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID WHERE T1.LName = 'Smith'
SELECT StuID, COUNT(PetID) FROM Has_Pet GROUP BY StuID
SELECT T1.StuID ,  COUNT(T2.PetID) FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID GROUP BY T1.StuID
SELECT T1.Fname ,  T1.Sex FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID GROUP BY T1.StuID HAVING COUNT(*)  >  1
SELECT Fname ,  Sex FROM Student WHERE StuID IN (SELECT StuID FROM Has_Pet GROUP BY StuID HAVING COUNT(*)  >  1)
SELECT T1.LName FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T3.PetType = 'cat' AND T3.pet_age = 3
SELECT T1.LName FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID  =  T2.StuID JOIN Pets AS T3 ON T2.PetID  =  T3.PetID WHERE T3.PetType  =  "cat" AND T3.pet_age  =  3
SELECT avg(Age) FROM Student WHERE StuID NOT IN (SELECT StuID FROM Has_Pet)
SELECT avg(T1.Age) FROM Student AS T1 WHERE T1.StuID NOT IN (SELECT T2.StuID FROM Student AS T2 JOIN Has_Pet AS T3 ON T2.StuID = T3.StuID)
SELECT COUNT(DISTINCT Continent) FROM continents
SELECT count(DISTINCT Continent) FROM continents
SELECT continents.ContId ,  continents.Continent ,  count(countries.CountryId) FROM continents JOIN countries ON continents.ContId  =  countries.Continent GROUP BY continents.ContId ,  continents.Continent
SELECT T1.ContId ,  T1.Continent ,  count(T2.CountryId) FROM continents AS T1 LEFT JOIN countries AS T2 ON T1.ContId  =  T2.Continent GROUP BY T1.ContId ,  T1.Continent
SELECT COUNT(*) FROM countries
SELECT count(DISTINCT CountryName) FROM countries
SELECT T1.FullName ,  T1.Id ,  COUNT(T2.ModelId) FROM car_makers AS T1 JOIN model_list AS T2 ON T1.Maker = T2.Maker GROUP BY T1.Id ,  T1.FullName
SELECT car_makers.Id ,  car_makers.FullName ,  COUNT(*) FROM car_makers JOIN model_list ON car_makers.Maker  =  model_list.Maker GROUP BY car_makers.Id ,  car_makers.FullName
SELECT T2.Model FROM cars_data AS T1 JOIN model_list AS T2 ON T1.Id  =  T2.ModelId WHERE T1.Horsepower  =  (SELECT MIN(Horsepower) FROM cars_data)
SELECT T1.Model FROM car_names AS T1 JOIN cars_data AS T2 ON T1.MakeId  =  T2.Id ORDER BY T2.Horsepower ASC LIMIT 1
SELECT model FROM car_names WHERE MakeId IN (SELECT Id FROM cars_data WHERE Weight < (SELECT avg(Weight) FROM cars_data))
SELECT T1.Model FROM model_list AS T1 JOIN car_names AS T2 ON T1.Model  =  T2.Model JOIN cars_data AS T3 ON T2.MakeId  =  T3.Id WHERE T3.Weight < (SELECT avg(Weight) FROM cars_data)
SELECT DISTINCT T1.Maker FROM car_makers AS T1 JOIN model_list AS T2 ON T1.Id  =  T2.Maker JOIN car_names AS T3 ON T2.Model  =  T3.Model JOIN cars_data AS T4 ON T3.MakeId  =  T4.Id WHERE T4.Year  =  1970
