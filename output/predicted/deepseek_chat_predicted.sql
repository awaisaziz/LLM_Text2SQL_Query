SELECT COUNT(*) FROM singer
SELECT COUNT(*) FROM singer
SELECT Name ,  Country ,  Age FROM singer ORDER BY Age DESC
SELECT Name ,  Country ,  Age FROM singer ORDER BY Age DESC
SELECT avg(Age) ,  min(Age) ,  max(Age) FROM singer WHERE Country  =  'France'
SELECT AVG(Age) ,  MIN(Age) ,  MAX(Age) FROM singer WHERE Country  =  'French'
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Age  =  (SELECT min(Age) FROM singer)
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Age  =  (SELECT min(Age) FROM singer)
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 JOIN (SELECT Singer_ID FROM singer ORDER BY Age ASC LIMIT 1) AS T2 ON T1.Singer_ID  =  T2.Singer_ID
SELECT T1.Song_Name ,  T1.Song_release_year FROM singer AS T1 WHERE T1.Singer_ID IN (SELECT Singer_ID FROM singer WHERE Age  =  (SELECT min(Age) FROM singer))
SELECT DISTINCT Country FROM singer WHERE Age > 20
SELECT DISTINCT Country FROM singer WHERE Age > 20
