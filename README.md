<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/AustinJamesWolff/housing_supply_and_demand">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Housing Supply and Demand Indicator</h3>

  <p align="center">
    Identify which markets have the most ideal supply/demand dynamics for investors.
    <br />
    <br />
    <a href="https://github.com/AustinJamesWolff/housing_supply_and_demand">Main Repo</a>
    ·
    <a href="https://github.com/AustinJamesWolff/housing_supply_and_demand/issues">Report Bug</a>
    ·
    <a href="https://github.com/AustinJamesWolff/housing_supply_and_demand/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
-->


<!-- ABOUT THE PROJECT -->
## About The Project

This repo is meant to help a real estate investor determine the best market to invest into multifamily properties, and therefore prioritizes datasets such as job growth, population growth, and rent growth.

The following questions will be answered:

1. Which cities have the most job growth over a 10-year period? And which cities have the **fastest** job growth?
2. Of the top 25 cities with the best job growth, which have the best rent-to-price ratio? (This could offer a better cash-on-cash return and potentially a superior IRR depending on appreciation.)
3. Which neighborhoods are:
   1. Having an increasing percent of renters, and of those,
   2. Which are experiencing the most rent growth?
4. Which neighborhoods are about to undergo gentrification in Los Angeles County?
   1. This will involve looking at the rent spread (between the 1st Quartile and 3rd Quartile), then turning that spread into its own metric, then training a linear autoregressive model on it, then using that model to predict which census tracts will undergo gentrification in 3 years.
   2. I am choosing Los Angeles County so I can drive to the neighborhoods predicted and view them for myself.
5. Which neihgborhoods have the most growth of same-sex households?
6. Is there a correlation (using a p-value of 0.05) between same-sex household growth and rent growth?
   1. If so, what neighborhoods are experiencing same-sex household growth that are predominantly renter-occupied, with rent growth?
      1. The thesis is, if rent growth is correlated with same-sex household growth, we may be able to identify rent-growth opportunities that have not yet occured if we can identiy neighborhoods with a growing number of LGBTQ individuals. 

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* Jupyter Notebook
* Python
* Pandas
* US Census (Census) API
* Bureau of Labor Statistics (BLS) API

### The Datasets Used

* Number of Jobs (BLS)
  * City level
* Rent Q1 (Census) (First Quartile)
  * Tract and City level
* Rent Q2 (Census) (Q2 is the Median)
  * Tract and City level
* Rent Q3 (Census) (Third Quartile)
  * Tract and City level
* Median Unit Price (Census)
  * Tract and City level
* Percent Renter-Occupied (Census)
  * Tract and City level
* Population (Census)
  * Tract and City level


### US Census Data Table Codes

#### At The Tract and City Level
* Median Rent
  * 2010-2021: B25064_001E
* Renter-Occupied Units
  * 2010-2014: DP04_0046E
  * 2015-2021: DP04_0047E
* Owner-Occupied Units
  * 2010-2014: DP04_0045E
  * 2015-2021: DP04_0046E
* Occupied Units
  * 2010-2021: DP04_0002E
* Total Housing Units
  * 2010-2021: DP04_0001E
* Vacant Units
  * 2010-2021: DP04_0003E
* Vacancy Rate
  * 2010-2021: DP04_0003PE
* Avg Owner Household Size
  * 2010-2014: DP04_0047E
  * 2015-2021: DP04_0048E
* Avg Renter Household Size
  * 2010-2014: DP04_0048E
  * 2015-2021: DP04_0049E
* Total Housing Units
  * 2010-2021: DP04_0001E
* Median Household Income
  * 2010-2021: B19013_001E
* Unemployment
  * 2010-2021: DP03_0009PE

#### At The Block Group Level

* Population
  * 2013-2021: B01003_001E
* Rent Distribution
  * 'B25063_003E': 'rent_less_than_100',
    'B25063_004E': 'rent_100_to_149',
    'B25063_005E': 'rent_150_to_199',
    'B25063_006E': 'rent_200_to_249',
    'B25063_007E': 'rent_250_to_299',
    'B25063_008E': 'rent_300_to_349',
    'B25063_009E': 'rent_350_to_399',
    'B25063_010E': 'rent_400_to_449',
    'B25063_011E': 'rent_450_to_499',
    'B25063_012E': 'rent_500_to_549',
    'B25063_013E': 'rent_550_to_599',
    'B25063_014E': 'rent_600_to_649',
    'B25063_015E': 'rent_650_to_699',
    'B25063_016E': 'rent_700_to_749',
    'B25063_017E': 'rent_750_to_799',
    'B25063_018E': 'rent_800_to_899',
    'B25063_019E': 'rent_900_to_999',
    'B25063_020E': 'rent_1000_to_1249',
    'B25063_021E': 'rent_1250_to_1449',
    'B25063_022E': 'rent_1500_to_1999',
    'B25063_023E': 'rent_2000_to_2499',
    'B25063_024E': 'rent_2500_to_2999',
    'B25063_025E': 'rent_3000_to_3499',
    'B25063_026E': 'rent_3500_or_more'
* Household Income Distribution
  * 'B19001_002E': 'income_less_than_10000',
    'B19001_003E': 'income_10000_to_14999',
    'B19001_004E': 'income_15000_to_19999',
    'B19001_005E': 'income_20000_to_24999',
    'B19001_006E': 'income_25000_to_29999',
    'B19001_007E': 'income_30000_to_34999',
    'B19001_008E': 'income_35000_to_39999',
    'B19001_009E': 'income_40000_to_44999',
    'B19001_010E': 'income_45000_to_49999',
    'B19001_011E': 'income_50000_to_59999',
    'B19001_012E': 'income_60000_to_74999',
    'B19001_013E': 'income_75000_to_99999',
    'B19001_014E': 'income_100000_to_124999',
    'B19001_015E': 'income_125000_to_149999',
    'B19001_016E': 'income_150000_to_199999',
    'B19001_017E': 'income_200000_or_more'
* 


### Visualizations Created

* Map of Rent-to-Price ratios by city
* Map of Rent-to-Price ratios for all tracts in a city

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/AustinJamesWolff/housing_supply_and_demand.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#top">back to top</a>)</p>
-->


<!-- USAGE EXAMPLES
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>
-->


<!-- ROADMAP
## Roadmap

- [] Feature 1
- [] Feature 2
- [] Feature 3
    - [] Nested Feature

See the [open issues](https://github.com/AustinJamesWolff/housing_supply_and_demand/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>
-->



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Austin Wolff - austinwolff1997@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p>
-->


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[contributors-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[forks-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/network/members
[stars-shield]: https://img.shields.io/github/stars/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[stars-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/stargazers
[issues-shield]: https://img.shields.io/github/issues/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[issues-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/issues
[license-shield]: https://img.shields.io/github/license/AustinJamesWolff/housing_supply_and_demand.svg?style=for-the-badge
[license-url]: https://github.com/AustinJamesWolff/housing_supply_and_demand/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
