# WeatherSetting



/
🔸 Sub-task 2.1
Get Monday's morning temperature in Ingolstadt.
💡 Expected output: 11 deg C

🔸 Sub-task 2.2
Get the days (indices) when Berlin had a morning temperature less than 10°C.
💡 Expected output:

lua
Copy
Edit
[[0]
 [2]
 [3]
 [6]]
🔸 Sub-task 2.3
Add 7-day temperature data of Munich (shape: (7, 3)) to the dataset.
Overwrite the existing array to include Munich as the fifth city.

The new shape should be (7, 5, 3)
💡 Expected output: Shown as new temperature data matrix with Munich added.

🔸 Sub-task 2.4
Calculate the daily average temperature of each city for the whole week.
💡 Expected output: A (7, 5) matrix, where each row = day and each column = city average temperature for that day.

🔸 Sub-task 2.5
Find the city (or cities) with the lowest average daily temperature over the whole week.

/