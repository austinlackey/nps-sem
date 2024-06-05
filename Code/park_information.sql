SELECT        Parks.ParkId AS park_id, Parks.UnitCode AS unit_code, Parks.Name AS name, ParkTypes.Name AS designation, PopulationCenters.Name AS population_center, Regions.Name AS region
FROM            Parks INNER JOIN
                         ParkTypes ON Parks.ParkTypeId = ParkTypes.ParkTypeId INNER JOIN
                         Regions ON Parks.RegionId = Regions.RegionId INNER JOIN
                         PopulationCenters ON Parks.PopulationCenterId = PopulationCenters.PopulationCenterId