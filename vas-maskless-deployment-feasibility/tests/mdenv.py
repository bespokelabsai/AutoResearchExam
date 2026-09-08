import numpy as np

from mdobs import (
    BLOCK_COAL,
    BLOCK_DOWN_LADDER,
    BLOCK_FLOOR,
    BLOCK_FURNACE,
    BLOCK_GEM,
    BLOCK_IRON,
    BLOCK_LAVA,
    BLOCK_PLANT,
    BLOCK_SAND,
    BLOCK_STONE,
    BLOCK_TABLE,
    BLOCK_TREE,
    BLOCK_UP_LADDER,
    BLOCK_VOID,
    BLOCK_WALL,
    BLOCK_WATER,
    FLAG_ITEM,
    FLAG_MELEE,
    FLAG_PASSIVE,
    FLAG_RANGED,
    MAX_STEPS,
    N_ACTIONS,
    build_obs,
)

GRID = 14
PAD = 2
N_FLOORS = 5
KILLS_TO_DESCEND = 8
MAX_MOBS = 3

A_NORTH, A_SOUTH, A_WEST, A_EAST = 0, 1, 2, 3
A_DO = 4
A_PLACE_STONE, A_PLACE_TABLE, A_PLACE_FURNACE, A_PLACE_PLANT = 5, 6, 7, 8
A_CRAFT_WOOD_PICK, A_CRAFT_STONE_PICK, A_CRAFT_IRON_PICK = 9, 10, 11
A_CRAFT_WOOD_SWORD, A_CRAFT_STONE_SWORD, A_CRAFT_IRON_SWORD = 12, 13, 14
A_CRAFT_ARMOR, A_CRAFT_AMULET = 15, 16
A_DRINK_POTION, A_DRINK_WATER = 17, 18
A_ENCHANT = 19
A_DESCEND, A_ASCEND, A_REST, A_NOOP = 20, 21, 22, 23

DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

WALKABLE = np.zeros(16, dtype=bool)
WALKABLE[[BLOCK_FLOOR, BLOCK_SAND, BLOCK_DOWN_LADDER, BLOCK_UP_LADDER]] = True

MOB_FLAG_ANY = FLAG_MELEE | FLAG_PASSIVE | FLAG_RANGED

ACHIEVEMENTS = (
    ("collect_wood", 1),
    ("place_table", 1),
    ("craft_wood_pickaxe", 1),
    ("craft_wood_sword", 1),
    ("collect_stone", 1),
    ("place_stone", 1),
    ("place_furnace", 1),
    ("place_plant", 1),
    ("defeat_slime", 1),
    ("drink_water", 1),
    ("craft_stone_pickaxe", 3),
    ("craft_stone_sword", 3),
    ("collect_coal", 3),
    ("reach_floor_2", 3),
    ("defeat_bat", 3),
    ("drink_potion", 3),
    ("collect_iron", 5),
    ("craft_iron_pickaxe", 5),
    ("craft_iron_sword", 5),
    ("reach_floor_3", 5),
    ("craft_armor", 5),
    ("defeat_golem", 5),
    ("collect_gem", 8),
    ("craft_gem_amulet", 8),
    ("reach_floor_4", 8),
    ("reach_floor_5", 8),
    ("enchant_sword", 8),
)
N_ACH = len(ACHIEVEMENTS)
ACH_REWARD = np.array([r for _, r in ACHIEVEMENTS], dtype=np.float32)
ACH_INDEX = {name: i for i, (name, _) in enumerate(ACHIEVEMENTS)}
MAX_RETURN = float(ACH_REWARD.sum())

RESOURCE_P = (
    (0.32, 0.30, 0.06, 0.00, 0.00),
    (0.16, 0.34, 0.10, 0.09, 0.00),
    (0.10, 0.32, 0.10, 0.13, 0.03),
    (0.08, 0.30, 0.08, 0.10, 0.09),
    (0.06, 0.30, 0.06, 0.08, 0.13),
)
RESOURCE_BLOCK = (BLOCK_TREE, BLOCK_STONE, BLOCK_COAL, BLOCK_IRON, BLOCK_GEM)

MINE_YIELD = {
    BLOCK_TREE: ("wood", 2, "collect_wood", 0),
    BLOCK_PLANT: ("wood", 1, "collect_wood", 0),
    BLOCK_STONE: ("stone", 2, "collect_stone", 1),
    BLOCK_COAL: ("coal", 1, "collect_coal", 1),
    BLOCK_IRON: ("iron", 1, "collect_iron", 2),
    BLOCK_GEM: ("gem", 1, "collect_gem", 3),
}

MOB_HP = (2, 3, 4, 5)
MOB_DAMAGE = (1, 1, 2, 2)
MOB_FLAG = (FLAG_MELEE, FLAG_RANGED, FLAG_MELEE, FLAG_MELEE)
MOB_ACH = ("defeat_slime", "defeat_bat", "defeat_golem", None)
FLOOR_MOBS = ((0,), (0, 1), (1, 2), (2, 3), (2, 3))


def _generate_floor(rng, level):
    grid = np.full((GRID, GRID), BLOCK_WALL, dtype=np.uint8)
    centres = []
    for _ in range(4):
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        y = int(rng.integers(1, GRID - 1 - h))
        x = int(rng.integers(1, GRID - 1 - w))
        grid[y : y + h, x : x + w] = BLOCK_FLOOR
        centres.append((y + h // 2, x + w // 2))
    for (y0, x0), (y1, x1) in zip(centres, centres[1:]):
        grid[y0, min(x0, x1) : max(x0, x1) + 1] = BLOCK_FLOOR
        grid[min(y0, y1) : max(y0, y1) + 1, x1] = BLOCK_FLOOR

    floor_mask = grid == BLOCK_FLOOR
    touching = np.zeros((GRID, GRID), dtype=bool)
    touching[1:, :] |= floor_mask[:-1, :]
    touching[:-1, :] |= floor_mask[1:, :]
    touching[:, 1:] |= floor_mask[:, :-1]
    touching[:, :-1] |= floor_mask[:, 1:]
    candidates = np.argwhere((grid == BLOCK_WALL) & touching)
    draws = rng.random(len(candidates))
    edges = np.cumsum(RESOURCE_P[level])
    for (y, x), d in zip(candidates, draws):
        for k, edge in enumerate(edges):
            if d < edge:
                grid[y, x] = RESOURCE_BLOCK[k]
                break

    open_cells = np.argwhere(grid == BLOCK_FLOOR)
    order = rng.permutation(len(open_cells))
    picks = [tuple(int(v) for v in open_cells[i]) for i in order]
    for cell in picks[:2]:
        grid[cell] = BLOCK_WATER
    if level >= 3:
        grid[picks[2]] = BLOCK_LAVA
    if level >= 1:
        grid[picks[3]] = BLOCK_SAND

    open_cells = np.argwhere(grid == BLOCK_FLOOR)
    order = rng.permutation(len(open_cells))
    shuffled = open_cells[order]
    start = tuple(int(v) for v in shuffled[0])
    dist = np.abs(shuffled - np.array(start)).sum(axis=1)
    if level < N_FLOORS - 1:
        ladder = tuple(int(v) for v in shuffled[int(np.argmax(dist))])
        grid[ladder] = BLOCK_DOWN_LADDER
    if level >= 1:
        grid[start] = BLOCK_UP_LADDER
    return grid, start


class MiniDungeon:
    """One MiniDungeon episode, fully determined by its map seed."""

    def __init__(self, map_seed):
        self.map_seed = int(map_seed)
        gen_rng = np.random.default_rng(np.random.SeedSequence([0xD1CE, self.map_seed]))
        self.grids = []
        self.starts = []
        for level in range(N_FLOORS):
            grid, start = _generate_floor(gen_rng, level)
            self.grids.append(grid)
            self.starts.append(start)
        self.rng = np.random.default_rng(np.random.SeedSequence([0xB0B, self.map_seed]))

        self.floor = 0
        self.pos = self.starts[0]
        self.t = 0
        self.done = False
        self.kills_on_floor = 0
        self.health = 10
        self.food = 10
        self.wood = 0
        self.stone = 0
        self.coal = 0
        self.iron = 0
        self.gem = 0
        self.potions = 0
        self.pickaxe = 0
        self.sword = 0
        self.armor = 0
        self.amulet = 0
        self.enchanted = 0
        self.achieved = np.zeros(N_ACH, dtype=bool)
        self.mobs = [[] for _ in range(N_FLOORS)]
        self.items = [set() for _ in range(N_FLOORS)]
        self._padded = np.full((GRID + 2 * PAD, GRID + 2 * PAD), BLOCK_VOID, dtype=np.uint8)
        self._flags = np.zeros((GRID + 2 * PAD, GRID + 2 * PAD), dtype=np.uint8)
        self._sync_padded()


    def _sync_padded(self):
        self._padded[PAD:-PAD, PAD:-PAD] = self.grids[self.floor]

    def _sync_flags(self):
        self._flags[:] = 0
        for m in self.mobs[self.floor]:
            self._flags[m[0] + PAD, m[1] + PAD] |= MOB_FLAG[m[2]]
        for (y, x) in self.items[self.floor]:
            self._flags[y + PAD, x + PAD] |= FLAG_ITEM

    def _in_bounds(self, y, x):
        return 0 <= y < GRID and 0 <= x < GRID

    def _block_at(self, y, x):
        if not self._in_bounds(y, x):
            return BLOCK_VOID
        return int(self.grids[self.floor][y, x])

    def _mob_at(self, y, x):
        for m in self.mobs[self.floor]:
            if m[0] == y and m[1] == x:
                return m
        return None

    def _neighbours(self):
        y, x = self.pos
        return [(y + dy, x + dx) for dy, dx in DELTA]

    def _nearby(self, block):
        """True if `block` occurs in the 3x3 neighbourhood of the player."""
        y, x = self.pos
        sub = self._padded[y + PAD - 1 : y + PAD + 2, x + PAD - 1 : x + PAD + 2]
        return bool((sub == block).any())

    def compact_state(self):
        """The (crop_block, crop_flags, extras_raw) triple mdobs.build_obs consumes."""
        y, x = self.pos
        self._sync_flags()
        crop_block = self._padded[y : y + 2 * PAD + 1, x : x + 2 * PAD + 1].reshape(-1).copy()
        crop_flags = self._flags[y : y + 2 * PAD + 1, x : x + 2 * PAD + 1].reshape(-1).copy()
        flags = self.armor | (self.amulet << 1) | (self.enchanted << 2)
        extras = np.array(
            [
                self.wood,
                self.stone,
                self.coal,
                self.iron,
                self.gem,
                self.pickaxe,
                self.sword,
                max(0, self.health),
                max(0, self.food),
                self.potions,
                flags,
                self.t,
            ],
            dtype=np.uint16,
        )
        return crop_block, crop_flags, extras

    def observation(self):
        return build_obs(*self.compact_state())


    def _do_target(self):
        """The neighbour `do` would act on, as (kind, cell), scanning N, S, W, E.

        kind is 'mob', 'item', 'mine' or None.
        """
        for cell in self._neighbours():
            y, x = cell
            if not self._in_bounds(y, x):
                continue
            if self._mob_at(y, x) is not None:
                return "mob", cell
        for cell in self._neighbours():
            if cell in self.items[self.floor]:
                return "item", cell
        for cell in self._neighbours():
            spec = MINE_YIELD.get(self._block_at(*cell))
            if spec is not None and self.pickaxe >= spec[3]:
                return "mine", cell
        return None, None

    def _place_target(self, allow_liquid=False):
        for cell in self._neighbours():
            b = self._block_at(*cell)
            if b in (BLOCK_FLOOR, BLOCK_SAND) and self._mob_at(*cell) is None:
                return cell
            if allow_liquid and b in (BLOCK_WATER, BLOCK_LAVA):
                return cell
        return None

    def _water_adjacent(self):
        return any(self._block_at(*cell) == BLOCK_WATER for cell in self._neighbours())


    def valid_mask(self):
        """Ground-truth action validity at the current state. Shape (24,) bool."""
        v = np.zeros(N_ACTIONS, dtype=bool)
        y, x = self.pos
        for a in range(4):
            dy, dx = DELTA[a]
            ny, nx = y + dy, x + dx
            v[a] = (
                self._in_bounds(ny, nx)
                and bool(WALKABLE[self._block_at(ny, nx)])
                and self._mob_at(ny, nx) is None
            )

        v[A_DO] = self._do_target()[0] is not None

        solid_target = self._place_target(allow_liquid=False) is not None
        v[A_PLACE_STONE] = self.stone >= 3 and self._place_target(allow_liquid=True) is not None
        v[A_PLACE_TABLE] = self.wood >= 4 and solid_target
        v[A_PLACE_FURNACE] = self.stone >= 4 and solid_target
        v[A_PLACE_PLANT] = self.wood >= 3 and solid_target

        table = self._nearby(BLOCK_TABLE)
        furnace = self._nearby(BLOCK_FURNACE)
        v[A_CRAFT_WOOD_PICK] = table and self.wood >= 1 and self.pickaxe == 0
        v[A_CRAFT_STONE_PICK] = table and self.wood >= 1 and self.stone >= 1 and self.pickaxe == 1
        v[A_CRAFT_IRON_PICK] = (
            table and furnace and self.wood >= 1 and self.coal >= 1 and self.iron >= 1 and self.pickaxe == 2
        )
        v[A_CRAFT_WOOD_SWORD] = table and self.wood >= 1 and self.sword == 0
        v[A_CRAFT_STONE_SWORD] = table and self.wood >= 1 and self.stone >= 1 and self.sword == 1
        v[A_CRAFT_IRON_SWORD] = (
            table and furnace and self.wood >= 1 and self.coal >= 1 and self.iron >= 1 and self.sword == 2
        )
        v[A_CRAFT_ARMOR] = table and furnace and self.iron >= 2 and self.coal >= 1 and self.armor == 0
        v[A_CRAFT_AMULET] = table and furnace and self.gem >= 1 and self.iron >= 1 and self.amulet == 0
        v[A_ENCHANT] = table and furnace and self.gem >= 1 and self.sword >= 3 and self.enchanted == 0

        v[A_DRINK_POTION] = self.potions >= 1 and self.health <= 6
        v[A_DRINK_WATER] = self.food <= 8 and self._water_adjacent()

        here = self._block_at(y, x)
        v[A_DESCEND] = (
            here == BLOCK_DOWN_LADDER
            and self.kills_on_floor >= KILLS_TO_DESCEND
            and self.floor < N_FLOORS - 1
        )
        v[A_ASCEND] = here == BLOCK_UP_LADDER and self.floor >= 1

        self._sync_flags()
        crop_flags = self._flags[y : y + 2 * PAD + 1, x : x + 2 * PAD + 1]
        v[A_REST] = self.health <= 6 and not bool((crop_flags & MOB_FLAG_ANY).any())
        v[A_NOOP] = True
        return v


    def _unlock(self, name):
        self.achieved[ACH_INDEX[name]] = True

    def _apply(self, action):
        if action < 4:
            dy, dx = DELTA[action]
            self.pos = (self.pos[0] + dy, self.pos[1] + dx)
            return
        if action == A_DO:
            kind, cell = self._do_target()
            if kind == "mob":
                mob = self._mob_at(*cell)
                mob[3] -= 1 + self.sword + self.enchanted
                if mob[3] <= 0:
                    self.mobs[self.floor].remove(mob)
                    self.kills_on_floor += 1
                    name = MOB_ACH[mob[2]]
                    if name is not None:
                        self._unlock(name)
                    if self.rng.random() < 0.5:
                        self.items[self.floor].add(cell)
            elif kind == "item":
                self.items[self.floor].discard(cell)
                self.potions += 1
            elif kind == "mine":
                field, amount, ach, _ = MINE_YIELD[self._block_at(*cell)]
                setattr(self, field, getattr(self, field) + amount)
                self._unlock(ach)
                self.grids[self.floor][cell] = BLOCK_FLOOR
                self._sync_padded()
            return
        if action == A_PLACE_STONE:
            self.stone -= 1
            self.grids[self.floor][self._place_target(allow_liquid=True)] = BLOCK_STONE
            self._unlock("place_stone")
            self._sync_padded()
            return
        if action in (A_PLACE_TABLE, A_PLACE_FURNACE, A_PLACE_PLANT):
            cell = self._place_target()
            if action == A_PLACE_TABLE:
                self.wood -= 4
                self.grids[self.floor][cell] = BLOCK_TABLE
                self._unlock("place_table")
            elif action == A_PLACE_FURNACE:
                self.stone -= 4
                self.grids[self.floor][cell] = BLOCK_FURNACE
                self._unlock("place_furnace")
            else:
                self.wood -= 1
                self.grids[self.floor][cell] = BLOCK_PLANT
                self._unlock("place_plant")
            self._sync_padded()
            return
        if action == A_CRAFT_WOOD_PICK:
            self.wood -= 1
            self.pickaxe = 1
            self._unlock("craft_wood_pickaxe")
        elif action == A_CRAFT_STONE_PICK:
            self.wood -= 1
            self.stone -= 1
            self.pickaxe = 2
            self._unlock("craft_stone_pickaxe")
        elif action == A_CRAFT_IRON_PICK:
            self.wood -= 1
            self.coal -= 1
            self.iron -= 1
            self.pickaxe = 3
            self._unlock("craft_iron_pickaxe")
        elif action == A_CRAFT_WOOD_SWORD:
            self.wood -= 1
            self.sword = 1
            self._unlock("craft_wood_sword")
        elif action == A_CRAFT_STONE_SWORD:
            self.wood -= 1
            self.stone -= 1
            self.sword = 2
            self._unlock("craft_stone_sword")
        elif action == A_CRAFT_IRON_SWORD:
            self.wood -= 1
            self.coal -= 1
            self.iron -= 1
            self.sword = 3
            self._unlock("craft_iron_sword")
        elif action == A_CRAFT_ARMOR:
            self.iron -= 2
            self.coal -= 1
            self.armor = 1
            self._unlock("craft_armor")
        elif action == A_CRAFT_AMULET:
            self.gem -= 1
            self.iron -= 1
            self.amulet = 1
            self._unlock("craft_gem_amulet")
        elif action == A_ENCHANT:
            self.gem -= 1
            self.enchanted = 1
            self._unlock("enchant_sword")
        elif action == A_DRINK_POTION:
            self.potions -= 1
            self.health = min(10, self.health + 3)
            self._unlock("drink_potion")
        elif action == A_DRINK_WATER:
            self.food = min(10, self.food + 4)
            self._unlock("drink_water")
        elif action == A_DESCEND:
            self.floor += 1
            self.pos = self.starts[self.floor]
            self.kills_on_floor = 0
            self._sync_padded()
            self._unlock("reach_floor_%d" % (self.floor + 1))
        elif action == A_ASCEND:
            self.floor -= 1
            self.pos = self.starts[self.floor]
            self.kills_on_floor = 0
            self._sync_padded()
        elif action == A_REST:
            self.health = min(10, self.health + 1)

    def _mob_phase(self):
        floor_mobs = self.mobs[self.floor]
        py, px = self.pos
        for m in floor_mobs:
            if m[4] > 0:
                m[4] -= 1
            if max(abs(py - m[0]), abs(px - m[1])) <= 1:
                if m[4] == 0:
                    self.health -= max(0, MOB_DAMAGE[m[2]] - self.armor)
                    m[4] = 4
                continue
            if self.rng.random() < 0.75:
                dy = int(np.sign(py - m[0]))
                dx = int(np.sign(px - m[1]))
                cand = []
                if dy != 0:
                    cand.append((m[0] + dy, m[1]))
                if dx != 0:
                    cand.append((m[0], m[1] + dx))
                for ny, nx in cand:
                    if (
                        self._in_bounds(ny, nx)
                        and WALKABLE[self._block_at(ny, nx)]
                        and (ny, nx) != self.pos
                        and self._mob_at(ny, nx) is None
                    ):
                        m[0], m[1] = ny, nx
                        break
        if len(floor_mobs) < MAX_MOBS and self.rng.random() < 0.22:
            open_cells = np.argwhere(WALKABLE[self.grids[self.floor]])
            if len(open_cells):
                pick = open_cells[self.rng.integers(len(open_cells))]
                sy, sx = int(pick[0]), int(pick[1])
                if max(abs(sy - py), abs(sx - px)) >= 3 and self._mob_at(sy, sx) is None:
                    kinds = FLOOR_MOBS[self.floor]
                    kind = int(kinds[self.rng.integers(len(kinds))])
                    floor_mobs.append([sy, sx, kind, MOB_HP[kind], 0])

    def step(self, action):
        """Apply `action`. An invalid action is a silent no-op: no reward, no penalty."""
        if self.done:
            return self.observation(), 0.0, True
        before = self.achieved.copy()
        if self.valid_mask()[int(action)]:
            self._apply(int(action))
        self._mob_phase()
        self.t += 1
        if self.t % 60 == 0:
            self.food = max(0, self.food - 1)
        if self.food == 0 and self.t % 10 == 0:
            self.health -= 1
        reward = float(ACH_REWARD[self.achieved & ~before].sum())
        if self.health <= 0 or self.t >= MAX_STEPS:
            self.done = True
        return self.observation(), reward, self.done
