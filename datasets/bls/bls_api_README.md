This README will help explain how to use the BLS API (I find the official documentation not user-friendly).

Here is their example of using Python to access the API:
https://www.bls.gov/developers/api_python.htm#python2

Here is the link to the main documentation page on the SeriesID, which what you specify to get the desired table you wish to download:
https://www.bls.gov/help/hlpforma.htm#CE


Here is an example of how the data tables are formatted via SeriesID:
	Series ID    CEU0800000003
	Positions       Value           Field Name
	1-2             CE              Prefix
	3               U               Seasonal Adjustment Code
	4-11		08000000	Supersector and Industry Codes
	12-13           03              Data Type Code

Here is an explanation on each of the parameters in the Naional Employment SeriesID:

1. Prefix
   1. National: CE
2. Seasonal Adjustment Code
   1. U - Not seasonally adjusted
   2. S - Seasonally adjusted
3. Super Sector Code
   1. See this link: https://download.bls.gov/pub/time.series/ce/ce.supersector
   2. Use supersector code "00" for all non-farm jobs
4. Industry Codes
   1. See this link: https://download.bls.gov/pub/time.series/ce/ce.industry
   2. Use the full industry code "00000000" for total non-farm jobs
5. Data Type Code
   1. See this link: https://download.bls.gov/pub/time.series/ce/ce.datatype
   2. Use code "01" for "ALL EMPLOYEES, THOUSANDS"

So the SeriesID for National Employment for all non-farm jobs would be:
CEU0000000001

**But we want employment by the city. Unfortunately, the best we can do is by the MSA. The following is an explanation of the SeriesID for MSA-level employment:**

Series ID    SMU19197802023800001
	Positions       Value           Field Name
	1-2             SM              Prefix
	3               U               Seasonal Adjustment Code
	4-5             19              State Code
	6-10            19780           Area Code
	11-18           20238000        SuperSector and Industry Code
	19-20           01             	Data Type Code

1. Prefix: SM
2. Seasonal Adjustment Code
   1. U - Not seasonally adjusted
   2. S - Seasonally adjusted
3. State Code
   1. Same FIPS state codes that the US Census uses.
   2. See this link for a list: https://download.bls.gov/pub/time.series/sm/sm.state
   3. We can use 99 for "All Metropolitan Statistical Areas"
4. Area Code
   1. See this link: https://download.bls.gov/pub/time.series/sm/sm.area
   2. We can use 99999 for "All Metropolitan Statistical Areas"
5. Super Sector Codes
   1. See this link: https://download.bls.gov/pub/time.series/sm/sm.supersector
   2. Use "00" for total non-farm jobs
6. Industry Codes
   1. Use "00000000" for total non-farm
7. Data Type Codes
   1. Use "01" for "All Employees, In Thousands"

This is the SeriesID for the Los Angeles Metro:
SMS06310800000000001

So the final SeriesID we may want to use is:
SMS99999990000000001

I want to use the CES (instead of the CPS) because the CES uses numbers submitted from businesses, whereas the CPS sues data submitted from households.

As a rule, use seasonally adjusted data when comparing months, and not seasonally adjusted when comparing years.
