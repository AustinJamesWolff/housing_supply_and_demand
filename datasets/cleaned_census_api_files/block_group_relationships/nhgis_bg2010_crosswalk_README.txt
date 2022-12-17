----------------------------------------------------------------------------------------------------
Documentation for NHGIS crosswalk files...
    from 2010 block groups...
    to 2020 block groups, census tracts, or counties
----------------------------------------------------------------------------------------------------

Contents
    - Data Summary
    - Zone Identifiers
    - Notes
    - Citation and Use

Additional documentation on NHGIS crosswalks is available at:
    https://www.nhgis.org/user-resources/geographic-crosswalks


----------------------------------------------------------------------------------------------------
Data Summary
----------------------------------------------------------------------------------------------------

Each NHGIS crosswalk file provides interpolation weights for allocating census counts from a 
specified set of source zones to a specified set of target zones.

Crosswalk file naming scheme:
    nhgis_[source level code][source year]_[target level code][target year]{_state FIPS code*}.csv

    * A state code suffix indicates the extent covered for a file that is limited to single state. 
      Such files may contain some source zones from neighboring states in cases where the Census 
      Bureau adjusted state boundary lines between censuses. Files with no state code suffix in 
      their names cover the entire U.S. and (for crosswalks from 2010 or 2020) Puerto Rico.

Geographic level codes:
    blk - Block
    bgp - Block group part (intersections between block groups, places, county subdivisions, etc.)
            - 1990 NHGIS level ID: blck_grp_598
            - 2000 NHGIS level ID: blck_grp_090
    bg  - Block group
    tr  - Census tract
    co  - County

Crosswalk file content:
    - The top row is a header row
    - Each subsequent row represents a spatial intersection (an "atom") between a source block group 
      and a target zone
    - Fields:
        bg2010gj:  NHGIS GISJOIN identifier for the source 2010 block group
        bg2010ge:  Census Bureau standard GEOID identifier for the source 2010 block group
        
        bg2020gj, tr2020gj OR co2020gj:   NHGIS GISJOIN identifier for the 2020 target zone
        bg2020ge, tr2020ge OR co2020ge:   Census Bureau standard GEOID identifier for the 2020 
	                                  target zone
                     
        wt_pop:    Interpolation weight, total population
                   (Expected proportion of source zone's population located in target zone)
        wt_adult:  Interpolation weight, persons age 18 and over
                   (Expected proportion of source zone's adult population located in target zone)
        wt_fam:    Interpolation weight, total families
                   (Expected proportion of source zone's families located in target zone)
        wt_hh:     Interpolation weight, total households
                   (Expected proportion of source zone's households located in target zone)
        wt_hu:     Interpolation weight, total housing units
                   (Expected proportion of source zone's housing units located in target zone)

        parea:     The portion of the source zone's land* area shared with target zone
                   * If a source zone's area includes no land, then this value is based on the
                     water area


----------------------------------------------------------------------------------------------------
Zone Identifiers
----------------------------------------------------------------------------------------------------

Block groups
    - GISJOIN: 15-character concatenation of:
        - "G"                 1 character
        - State NHGIS code:   3 digits (FIPS + "0")
        - County NHGIS code:  4 digits (FIPS + "0")
        - Census tract code:  6 digits
        - Block group code:   1 digit
    - GEOID: 12-character concatenation of:
        - State FIPS code:    2 digits
        - County FIPS code:   3 digits
        - Census tract code:  6 digits
        - Block group code:   1 digit

Census tracts
    - GISJOIN: 14-character concatenation of:
        - "G"                 1 character
        - State NHGIS code:   3 digits (FIPS + "0")
        - County NHGIS code:  4 digits (FIPS + "0")
        - Census tract code:  6 digits
    - GEOID: 11-character concatenation of:
        - State FIPS code:    2 digits
        - County FIPS code:   3 digits
        - Census tract code:  6 digits

Counties
    - GISJOIN: 8-character concatenation of:
        - "G"                 1 character
        - State NHGIS code:   3 digits (FIPS + "0")
        - County NHGIS code:  4 digits (FIPS + "0")
    - GEOID (co2010ge): 5-character concatenation of:
        - State FIPS code:    2 digits
        - County FIPS code:   3 digits


----------------------------------------------------------------------------------------------------
Notes
----------------------------------------------------------------------------------------------------

* For documentation of the models and procedures used to derive the crosswalks, see 
    https://www.nhgis.org/user-resources/geographic-crosswalks


----------------------------------------------------------------------------------------------------
Citation and Use
----------------------------------------------------------------------------------------------------

All persons are granted a limited license to use this documentation and the accompanying data, 
subject to the following conditions:

* REDISTRIBUTION: You will not redistribute the data without permission.

  You may publish a subset of the data to meet journal requirements for accessing data related to 
  a particular publication. Contact us (nhgis@umn.edu) for permission for any other redistribution;
  we will consider requests for free and commercial redistribution.

* CITATION: You will cite NHGIS appropriately.

    * Publications and research reports employing NHGIS data should include the following citation:

        Steven Manson, Jonathan Schroeder, David Van Riper, Tracy Kugler, and Steven Ruggles.
        IPUMS National Historical Geographic Information System: Version 16.0 [dataset].
        Minneapolis, MN: IPUMS. 2021. http://doi.org/10.18128/D050.V16.0

    * For policy briefs, online resources, or articles in the popular press, we recommend that you
      cite the use of NHGIS data as follows:

        IPUMS NHGIS, University of Minnesota, www.nhgis.org.

These terms of use are a legally binding agreement. You can use the data only in accordance with 
these terms, and any other use is a violation of the agreement. Violations may result in revocation
of the agreement and prohibition from using other IPUMS data. If IPUMS or our partners are harmed 
from your violation, you are responsible for all damages, including reasonable attorney's fees and 
expenses.

In addition, we request that users send us a copy of any publications, research reports, or 
educational material making use of the data or documentation.

Send electronic material to: nhgis@umn.edu

Printed matter should be sent to:

    IPUMS NHGIS
    Institute for Social Research and Data Innovation
    University of Minnesota
    50 Willey Hall
    225 19th Ave S
    Minneapolis, MN 55455

