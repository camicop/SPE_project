# Basic Traffic Light Intersection Simulation

This project simulates a basic `+`-shaped intersection in SUMO, where vehicles move only **north-south** and **east-west** (and vice versa), regulated by a **static traffic light** with green and yellow phases.

---

## Project Structure

| File | Description |
|------|-------------|
| `nodes.nod.xml` | Defines the nodes (intersections or road endpoints). |
| `edges.edg.xml` | Describes the roads (edges) connecting the nodes. |
| `connections.con.xml` | Specifies allowed movements between edges at the intersection. |
| `trafficlight.tll.xml` | Sets the traffic light logic, including green/yellow phases. |
| `routes.rou.xml` | Contains vehicle types, routes, and traffic flows. |
| `config.sumocfg` | The main configuration file for running the simulation. |

---

## Traffic Light Logic

The intersection uses a **4-phase static traffic light**:

1. **North-South green** — 30 seconds  
2. **North-South yellow** — 3 seconds  
3. **East-West green** — 30 seconds  
4. **East-West yellow** — 3 seconds

This ensures that only one direction moves at a time, avoiding collisions.

---

## Traffic Flows

The file `routes.rou.xml` defines vehicle flows with the following directions:

- **North → South**
- **South → North**
- **West → East**
- **East → West**

Each flow starts at second `0`, ends at `600` (10 minutes), and spawns vehicles at a rate of `600 vehicles/hour`.

---

## Running the Simulation

Once all files are in place:
```
netconvert -n nodes.nod.xml -e edges.edg.xml -x connections.con.xml -o simple_intersection.net.xml
sumo-gui -c config.sumocfg
```