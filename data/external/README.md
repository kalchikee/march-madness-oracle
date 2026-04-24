# External reference data

Commit small CSVs here for data that rarely changes but the pipeline depends on.

| File | Content | Source |
|------|---------|--------|
| `conferences.csv` | `season,team,conference` | Manual / Wikipedia crosswalk |
| `coaches.csv` | `season,team,coach_name` | Sports-Reference coach pages |
| `arena_locations.csv` | `team,lat,lon,tz_offset` | Manual one-time build |
| `venues.csv` | `venue_id,venue_name,city,lat,lon,tz_offset` | Official tournament site list |

For the travel-distance feature (`src/madness/features/travel.py`) you need
both `arena_locations.csv` (team home campuses) and `venues.csv` (tournament
pod / region / Final Four locations).
