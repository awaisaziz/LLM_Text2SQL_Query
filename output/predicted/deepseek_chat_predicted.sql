SELECT COUNT(*) FROM singer
SELECT COUNT(*) FROM singer
SELECT Name ,  Country ,  Age FROM singer ORDER BY Age DESC
SELECT Name ,  Country ,  Age FROM singer ORDER BY Age DESC
SELECT avg(Age) ,  min(Age) ,  max(Age) FROM singer WHERE Country  =  'France'
SELECT avg(Age) ,  min(Age) ,  max(Age) FROM singer WHERE Country  =  'French'
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Age  =  (SELECT min(Age) FROM singer)
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 ORDER BY T1.Age ASC LIMIT 1
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Age  <=  ALL (SELECT Age FROM singer)
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Age  =  (SELECT min(Age) FROM singer)
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 ORDER BY T1.Age ASC LIMIT 1
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 JOIN (SELECT min(Age) AS min_age FROM singer) AS T2 ON T1.Age  =  T2.min_age
SELECT DISTINCT Country FROM singer WHERE Age > 20
SELECT DISTINCT Country FROM singer WHERE Age > 20
