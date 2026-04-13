--Identify which weeks have the highest and lowest performance
--Total sales
(SELECT week_start_date,
	SUM(weekly_sales) total_sales
FROM walmart_sales
GROUP BY week_start_date
ORDER BY total_sales DESC
LIMIT 5)

UNION ALL 

(SELECT week_start_date,
	SUM(weekly_sales) total_sales
FROM walmart_sales
GROUP BY week_start_date
ORDER BY total_sales ASC
LIMIT 5)

ORDER BY total_sales DESC;

--Average sales
(SELECT week_start_date,
	AVG(weekly_sales) average_sales
FROM walmart_sales
GROUP BY week_start_date
ORDER BY average_sales DESC
LIMIT 5)

UNION ALL 

(SELECT week_start_date,
	AVG(weekly_sales) average_sales
FROM walmart_sales
GROUP BY week_start_date
ORDER BY average_sales ASC
LIMIT 5)

ORDER BY average_sales DESC;

--Median Week
WITH total_sales AS (
SELECT week_start_date,
	SUM(weekly_sales) totals
FROM walmart_sales
GROUP BY week_start_date
),

median_week AS (
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY totals) median_sales
FROM total_sales
)

SELECT (
	SELECT week_start_date
	FROM total_sales
	WHERE totals = (SELECT median_sales FROM median_week)  
),
median_sales
FROM median_week;


--Find which stores have the largest gain between holiday and non-holiday weeks
--A negative number means there is a loss of sales in holiday weeks
WITH hday_means AS (
	SELECT store,
		AVG(CASE WHEN holiday_flag = 1 THEN weekly_sales END) AS hday_avg_sales,
		AVG(CASE WHEN holiday_flag = 0 THEN weekly_sales END) AS nonhday_avg_sales
	FROM walmart_sales
	GROUP BY store
)

(SELECT store, 
	hday_avg_sales - nonhday_avg_sales AS hday_sales_diff
FROM hday_means
ORDER BY hday_sales_diff DESC
LIMIT 5)

UNION ALL

(SELECT store, 
	hday_avg_sales - nonhday_avg_sales AS hday_sales_diff
FROM hday_means
ORDER BY hday_sales_diff ASC
LIMIT 5)

ORDER BY hday_sales_diff DESC;

--All stores by holiday uplift
WITH hday_means AS (
	SELECT store,
		AVG(CASE WHEN holiday_flag = 1 THEN weekly_sales END) AS hday_avg_sales,
		AVG(CASE WHEN holiday_flag = 0 THEN weekly_sales END) AS nonhday_avg_sales
	FROM walmart_sales
	GROUP BY store
)

SELECT store, 
	hday_avg_sales - nonhday_avg_sales AS hday_sales_diff
FROM hday_means
ORDER BY hday_sales_diff DESC;

