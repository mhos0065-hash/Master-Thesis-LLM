
source("0_setup.R")


pop <- tibble(
  borough = c(
    "Barking and Dagenham", "Barnet", "Bexley", "Brent",
    "Bromley", "Camden", "City of London", "Croydon",
    "Ealing", "Enfield", "Greenwich", "Hackney",
    "Hammersmith and Fulham", "Haringey", "Harrow", "Havering",
    "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea",
    "Kingston upon Thames", "Lambeth", "Lewisham", "Merton",
    "Newham", "Redbridge", "Richmond upon Thames", "Southwark",
    "Sutton", "Tower Hamlets", "Waltham Forest", "Wandsworth",
    "Westminster"
  ),
  population_2021 = c(
    218484, 395869, 249570, 335186,
    331096, 265547, 8600,  400731,
    349048, 337000, 291532, 281120,
    185426, 267169, 260476, 261693,
    309404, 278970, 245889, 156197,
    177507, 324243, 305842, 208660,
    355177, 310400, 197339, 317256,
    206349, 319652, 279840, 329677,
    261723
  )
)

# ── Helper: standardise borough names ───────────────────────
clean_borough <- function(x) {
  x %>%
    # Remove common prefixes
    str_remove_all("^(LB |RB |London Borough of |Royal Borough of )") %>%
    str_remove_all(" Corporation$| London Borough$") %>%
    # Replace & with "and"
    str_replace_all("&", "and") %>%
    # Convert to title case (but we'll fix exceptions after)
    str_to_title() %>%
    # Fix specific words that should be lower case (except first word)
    str_replace("\\bAnd\\b", "and") %>%
    str_replace("\\bUpon\\b", "upon") %>%
    str_replace("\\bOf\\b", "of") %>%
    str_replace("\\bThe\\b", "the") %>%
    # Trim any remaining whitespace
    str_trim()
}


# DataFrame :  aq_raw_23_anual, aq_raw_23_monthly, aq_raw_22_anual, aq_raw_22_monthly, aq_raw_21_anual, aq_raw_21_monthly
#             gas_raw_23, gas_raw_22, gas_raw_21 
#             energy_LEGGI_raw_23, GHG_LEGGI_raw_23, transport_LEGGI_raw_23
#             energy_LEGGI_raw_22, GHG_LEGGI_raw_22, transport_LEGGI_raw_22
#             energy_LEGGI_raw_21, GHG_LEGGI_raw_21, transport_LEGGI_raw_21
#             crime_raw_all, crime_raw_2023, crime_raw_2022, crime_raw_2021
#             waste_raw_23, waste_raw_22, waste_raw_21
#             collision_with_borough_23



# ── A: Air Quality — NO2 ────────────────────────────────────
# File: DT_data_2023.xlsx (adjust sheet/column names to match actual file)

aq_raw_23_anual <- read_excel("2023/DT_data_2023.xlsx", sheet = "Summary")
  
# Apply cleaning to the borough column
aq_raw_23_anual$borough <- clean_borough(aq_raw_23_anual$Borough)

# Verify the cleaned names
aq_raw_23_anual$borough

# Check which boroughs in aq_raw_23 are NOT in pop
missing <- setdiff(unique(aq_raw_23_anual$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

aq_raw_23_monthly <- read_excel("2023/DT_data_2023.xlsx", sheet = "Monthly summary")

# Apply cleaning to the borough column
aq_raw_23_monthly$borough <- clean_borough(aq_raw_23_monthly$Borough)

# Verify the cleaned names
aq_raw_23_monthly$borough

# Check which boroughs in aq_raw_23 are NOT in pop
missing <- setdiff(unique(aq_raw_23_monthly$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}


# File: DT_data_2022.xlsx (adjust sheet/column names to match actual file)
aq_raw_22_anual <- read_excel("2022/DT_data_2022.xlsx", sheet = "Summary")

# Apply cleaning to the borough column
aq_raw_22_anual$borough <- clean_borough(aq_raw_22_anual$Borough)

# Verify the cleaned names
aq_raw_22_anual$borough

# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(aq_raw_22_anual$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

aq_raw_22_monthly <- read_excel("2022/DT_data_2022.xlsx", sheet = "Monthly summary")

# Apply cleaning to the borough column
aq_raw_22_monthly$borough <- clean_borough(aq_raw_22_monthly$Borough)

# Verify the cleaned names
aq_raw_22_monthly$borough

# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(aq_raw_22_monthly$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

# File: DT_data_2021.xlsx (adjust sheet/column names to match actual file)
aq_raw_21_anual <- read_excel("2021/DT_data_2021.xlsx", sheet = "Summary")

# Apply cleaning to the borough column
aq_raw_21_anual$borough <- clean_borough(aq_raw_21_anual$Borough)

# Verify the cleaned names
aq_raw_21_anual$borough

# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(aq_raw_21_anual$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

aq_raw_21_monthly <- read_excel("2021/DT_data_2021.xlsx", sheet = "Monthly summary")

# Apply cleaning to the borough column
aq_raw_21_monthly$borough <- clean_borough(aq_raw_21_monthly$Borough)

# Verify the cleaned names
aq_raw_21_monthly$borough

# Check which boroughs in aq_raw_23 are NOT in pop
missing <- setdiff(unique(aq_raw_21_monthly$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}


# File: gas-consumption.xlsx (adjust sheet/column names to match actual file)
gas_raw_23 <- read_excel("2023/gas-consumption.xlsx", sheet = "2023")


# Apply cleaning to the borough column
gas_raw_23$borough <- clean_borough(gas_raw_23$Borough)

# Verify the cleaned names
gas_raw_23$borough

# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(gas_raw_23$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

gas_raw_22 <- read_excel("2023/gas-consumption.xlsx", sheet = "2022")


# Apply cleaning to the borough column
gas_raw_22$borough <- clean_borough(gas_raw_22$Borough)

# Verify the cleaned names
gas_raw_22$borough

# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(gas_raw_22$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

gas_raw_21 <- read_excel("2023/gas-consumption.xlsx", sheet = "2021")

# Apply cleaning to the borough column
gas_raw_21$borough <- clean_borough(gas_raw_21$Borough)

# Verify the cleaned names
gas_raw_21$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(gas_raw_21$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

energy_LEGGI_raw_23 <- read_excel("2023/LEGGI 2023_used.xlsx", sheet = "Borough KWH")

# Apply cleaning to the borough column
energy_LEGGI_raw_23$borough <- clean_borough(energy_LEGGI_raw_23$Borough)

# Verify the cleaned names
energy_LEGGI_raw_23$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(energy_LEGGI_raw_23$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

GHG_LEGGI_raw_23 <- read_excel("2023/LEGGI 2023_used.xlsx", sheet = "Borough GHG")

# Apply cleaning to the borough column
GHG_LEGGI_raw_23$borough <- clean_borough(GHG_LEGGI_raw_23$Borough)

# Verify the cleaned names
GHG_LEGGI_raw_23$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(GHG_LEGGI_raw_23$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

transport_LEGGI_raw_23 <- read_excel("2023/LEGGI 2023_used.xlsx", sheet = "Borough Transport")

# Apply cleaning to the borough column
transport_LEGGI_raw_23$borough <- clean_borough(transport_LEGGI_raw_23$Borough)

# Verify the cleaned names
transport_LEGGI_raw_23$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(transport_LEGGI_raw_23$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}


energy_LEGGI_raw_22 <- read_excel("2022/LEGGI 2022_used.xlsx", sheet = "Borough KWH")

# Apply cleaning to the borough column
energy_LEGGI_raw_22$borough <- clean_borough(energy_LEGGI_raw_22$Borough)

# Verify the cleaned names
energy_LEGGI_raw_22$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(energy_LEGGI_raw_22$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

GHG_LEGGI_raw_22 <- read_excel("2022/LEGGI 2022_used.xlsx", sheet = "Borough GHG")

# Apply cleaning to the borough column
GHG_LEGGI_raw_22$borough <- clean_borough(GHG_LEGGI_raw_22$Borough)

# Verify the cleaned names
GHG_LEGGI_raw_22$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(GHG_LEGGI_raw_22$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

transport_LEGGI_raw_22 <- read_excel("2022/LEGGI 2022_used.xlsx", sheet = "Borough Transport")

# Apply cleaning to the borough column
transport_LEGGI_raw_22$borough <- clean_borough(transport_LEGGI_raw_22$Borough)

# Verify the cleaned names
transport_LEGGI_raw_22$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(transport_LEGGI_raw_22$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

energy_LEGGI_raw_21 <- read_excel("2021/LEGGI 2021_used.xlsx", sheet = "Borough KWH")

# Apply cleaning to the borough column
energy_LEGGI_raw_21$borough <- clean_borough(energy_LEGGI_raw_21$Borough)

# Verify the cleaned names
energy_LEGGI_raw_21$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(energy_LEGGI_raw_21$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

GHG_LEGGI_raw_21 <- read_excel("2021/LEGGI 2021_used.xlsx", sheet = "Borough GHG")

# Apply cleaning to the borough column
GHG_LEGGI_raw_21$borough <- clean_borough(GHG_LEGGI_raw_21$Borough)

# Verify the cleaned names
GHG_LEGGI_raw_22$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(GHG_LEGGI_raw_21$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

transport_LEGGI_raw_21 <- read_excel("2021/LEGGI 2021_used.xlsx", sheet = "Borough Transport")

# Apply cleaning to the borough column
transport_LEGGI_raw_21$borough <- clean_borough(transport_LEGGI_raw_21$Borough)

# Verify the cleaned names
transport_LEGGI_raw_21$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(transport_LEGGI_raw_21$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

crime_raw_all <- read_excel("2023/MPS Crime.xlsx")

# Apply cleaning to the borough column
crime_raw_all$borough <- clean_borough(crime_raw_all$Borough)

# Verify the cleaned names
crime_raw_all$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(crime_raw_all$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}




crime_raw_2023 <- crime_raw_all %>%
  select(borough, `crime_per_1k_2023`)

crime_raw_2022 <- crime_raw_all %>%
  select(borough, `crime_per_1k_2022`)

crime_raw_2021 <- crime_raw_all %>%
  select(borough, `crime_per_1k_2021`)

transport_LEGGI_raw_21 <- read_excel("2021/LEGGI 2021_used.xlsx", sheet = "Borough Transport")

# Apply cleaning to the borough column
transport_LEGGI_raw_21$borough <- clean_borough(transport_LEGGI_raw_21$Borough)

# Verify the cleaned names
transport_LEGGI_raw_21$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(transport_LEGGI_raw_21$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

crime_raw_all <- read_csv("2023/MPS Borough Level Crime (Historical).csv")

# Apply cleaning to the borough column
crime_raw_all$borough <- clean_borough(crime_raw_all$Borough)

# Verify the cleaned names
crime_raw_all$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(crime_raw_all$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

waste_raw_23 <- read_excel("2023/2023 Waste Data Interrogator.xlsx", sheet = "Borough (Waste)")

# Apply cleaning to the borough column
waste_raw_23$borough <- clean_borough(waste_raw_23$Borough)

# Verify the cleaned names
waste_raw_23$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(waste_raw_23$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

waste_raw_22 <- read_excel("2022/2022 Waste Data Interrogator.xlsx", sheet = "Borough (Waste)")

# Apply cleaning to the borough column
waste_raw_22$borough <- clean_borough(waste_raw_22$Borough)

# Verify the cleaned names
waste_raw_22$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(waste_raw_22$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}

waste_raw_21<- read_excel("2021/2021 Waste Data Interrogator.xlsx", sheet = "Borough (Waste)")

# Apply cleaning to the borough column
waste_raw_21$borough <- clean_borough(waste_raw_21$Borough)

# Verify the cleaned names
waste_raw_21$borough


# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(waste_raw_21$borough), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}


# ── Step 1: Load London borough boundaries from GitHub ───────
boroughs_sf <- st_read(
  "https://github.com/LingruFeng/GIS_assessment/blob/main/London_Boroughs.gpkg?raw=true",
  quiet = TRUE
) %>%
  st_transform(4326)   # reproject to WGS84 to match collision lat/lon

# Verify
cat("Number of boroughs:", nrow(boroughs_sf), "\n")
print(boroughs_sf$name)
cat("CRS:", st_crs(boroughs_sf)$epsg, "\n")

# Check which boroughs in aq_raw_22 are NOT in pop
missing <- setdiff(unique(boroughs_sf$name), pop$borough)
if (length(missing) > 0) {
  print("These boroughs are missing from pop:")
  print(missing)
} else {
  print("All borough names match perfectly!")
}


collission_raw_23 <- read_csv("2023/dft-road-casualty-statistics-collision-2023.csv")

names(collission_raw_23)

collission_raw_23 <- collission_raw_23 %>%
  filter(!is.na(latitude), !is.na(longitude))

# ── Step 4: Convert collision data to sf object ──────────────
collision_sf_23 <- collission_raw_23 %>%
  st_as_sf(
    coords = c("longitude", "latitude"),
    crs    = 4326        # WGS84 — matches collision lat/lon columns
  )

# ── Step 5: Confirm CRS alignment ────────────────────────────
cat("Collision CRS:", st_crs(collision_sf_23)$epsg,  "\n")
cat("Borough CRS:  ", st_crs(collision_sf_23)$epsg,   "\n")

# Reproject if CRS do not match (safety check)
if (!st_crs(collision_sf_23) == st_crs(collision_sf_23)) {
  collision_sf_23 <- st_transform(collision_sf_23, st_crs(collision_sf_23))
  cat("CRS transformed to match borough boundaries.\n")
} else {
  cat("CRS aligned — no transformation needed.\n")
}

# After spatial join
collision_with_borough_23 <- st_join(
  collision_sf_23,
  boroughs_sf %>% select(name),
  join = st_within
)

# Diagnostics
n_total     <- nrow(collision_with_borough_23)
n_unmatched <- sum(is.na(collision_with_borough_23$name))
n_matched   <- n_total - n_unmatched

cat("Total collision records:               ", n_total,     "\n")
cat("Matched to a borough:                  ", n_matched,   "\n")
cat("Unmatched (outside borough boundaries):", n_unmatched, "\n")
cat("Percentage unmatched:                  ",
    round(n_unmatched / n_total * 100, 3), "%\n")

collision_with_borough_23 <- collision_with_borough_23 %>%
  filter(!is.na(name)) %>%
  st_drop_geometry()      # geometry no longer needed after join

cat("Final collision rows retained:", nrow(collision_with_borough_23), "\n")

# ── Step 9: Validate borough names ───────────────────────────
# Check all borough names in collision data exist in master panel
unexpected <- collision_with_borough_23 %>%
  filter(!name %in% boroughs_sf$name) %>%
  distinct(name)

if (nrow(unexpected) > 0) {
  cat("WARNING — unexpected borough names:\n")
  print(unexpected)
} else {
  cat("All borough names validated successfully.\n")
}


# Show top 10 matched boroughs
first_100 <- collision_with_borough_23 %>%
head(100) %>%
  count(name, sort = TRUE) %>%
  print(n = 10)

# Change the column name from name to borough
collision_with_borough_23 <- collision_with_borough_23 %>%
  rename(borough = name)

names(collision_with_borough_23)

first_100 <- collision_with_borough_23 %>%
  head(100) %>%
  st_drop_geometry() %>%          # remove geometry column
  select(collision_index, borough)   # adjust column names if different

print(first_100)


