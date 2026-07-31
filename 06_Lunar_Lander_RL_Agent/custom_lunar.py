import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import (
    LunarLander, VIEWPORT_W, VIEWPORT_H, SCALE, LANDER_POLY, LEG_AWAY, LEG_W, LEG_H, LEG_DOWN,
    LEG_SPRING_TORQUE, INITIAL_RANDOM, ContactDetector
)
import Box2D
from Box2D.b2 import edgeShape, fixtureDef, polygonShape, revoluteJointDef, circleShape

def safe_color(r, g, b, a=None):
    """Ensure color channels are strictly integers in the range [0, 255]."""
    r_c = max(0, min(255, int(r)))
    g_c = max(0, min(255, int(g)))
    b_c = max(0, min(255, int(b)))
    if a is not None:
        a_c = max(0, min(255, int(a)))
        return (r_c, g_c, b_c, a_c)
    return (r_c, g_c, b_c)

class CustomStyledLunarLander(LunarLander):
    """Custom LunarLander environment with flat moon surface, 2 blue lines, white ship, and red particles."""
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._destroy()

        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Flat Moon Terrain
        CHUNKS = 11
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        
        # Completely FLAT moon surface
        smooth_y = [self.helipad_y] * CHUNKS

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0, 0, 0)
        self.moon.color2 = (0, 0, 0)

        # Create WHITE Lander body (instead of purple)
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )
        # White body with light grey outline
        self.lander.color1 = (255, 255, 255)
        self.lander.color2 = (180, 180, 180)

        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        if self.enable_wind:
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        # Create WHITE Lander Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (255, 255, 255)
            leg.color2 = (180, 180, 180)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}

    def render(self):
        if self.render_mode is None:
            return

        import pygame
        from pygame import gfxdraw

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        # RED Engine Particles (clamped strictly between 0 and 255)
        for obj in self.particles:
            obj.ttl -= 0.15
            r = min(255, max(0, int((0.3 + 0.7 * min(1.0, max(0.0, obj.ttl))) * 255)))
            g = min(255, max(0, int(0.15 * max(0.0, obj.ttl) * 255)))
            b = min(255, max(0, int(0.15 * max(0.0, obj.ttl) * 255)))
            obj.color1 = (r, g, b)
            obj.color2 = (255, 30, 30)

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            c1 = safe_color(*getattr(obj, 'color1', (255, 255, 255)))
            c2 = safe_color(*getattr(obj, 'color2', (180, 180, 180)))
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=c1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=c2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=c1, points=path)
                    gfxdraw.aapolygon(self.surf, path, c1)
                    pygame.draw.aalines(
                        self.surf, color=c2, points=path, closed=True
                    )

        # 2 BLUE LINES marking landing pad (replacing yellow flags)
        for x in [self.helipad_x1, self.helipad_x2]:
            x = x * SCALE
            flagy1 = self.helipad_y * SCALE
            flagy2 = flagy1 + 45
            # Bold Electric Blue Lines
            pygame.draw.line(
                self.surf,
                color=(0, 180, 255),
                start_pos=(x, flagy1),
                end_pos=(x, flagy2),
                width=3,
            )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )
