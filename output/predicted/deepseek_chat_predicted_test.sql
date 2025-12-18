SELECT COUNT(*) FROM singer
SELECT COUNT(DISTINCT Singer_ID) FROM singer
SELECT Name ,  Country ,  Age FROM singer ORDER BY Age DESC
SELECT Name ,  Country ,  Age FROM singer ORDER BY Age DESC
SELECT avg(Age) ,  min(Age) ,  max(Age) FROM singer WHERE Country = 'France'
SELECT avg(T1.Age) ,  min(T1.Age) ,  max(T1.Age) FROM singer AS T1 WHERE T1.Country  =  'French'
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Age = (SELECT min(Age) FROM singer)
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Age = (SELECT min(Age) FROM singer)
SELECT DISTINCT T1.Country FROM singer AS T1 WHERE T1.Age > 20
SELECT DISTINCT Country FROM singer WHERE Age > 20
