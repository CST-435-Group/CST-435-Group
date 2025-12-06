"""
Player class with physics and controls
Python translation of launcher/frontend/src/components/rl/Player.js
"""

import math


class Player:
    """
    Player character for the platformer game.
    Handles physics, movement, and collision detection.
    """

    def __init__(self, x, y, width=32, height=32):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocityX = 0
        self.velocityY = 0
        self.speed = 5
        self.jumpPower = 15
        self.gravity = 0.8
        self.maxFallSpeed = 20
        self.isOnGround = False
        self.isAlive = True
        self.distance = 0
        self.score = 0
        self.isSprinting = False
        self.isDucking = False
        self.onIcePlatform = False  # Track if on ice (slippery)

    def update(self, platforms, deltaTime=1):
        """
        Update player physics and collision.

        Args:
            platforms: List of platform dictionaries
            deltaTime: Time delta (not used, kept for compatibility)
        """
        if not self.isAlive:
            return

        # Apply gravity
        self.velocityY += self.gravity
        if self.velocityY > self.maxFallSpeed:
            self.velocityY = self.maxFallSpeed

        # Apply velocity
        self.x += self.velocityX
        self.y += self.velocityY

        # Check ground collision
        self.isOnGround = False
        self.onIcePlatform = False
        for platform in platforms:
            if self.checkPlatformCollision(platform):
                self.isOnGround = True
                self.velocityY = 0
                self.y = platform['y'] - self.height
                # Check if platform is ice (slippery)
                if platform.get('type') == 'ice':
                    self.onIcePlatform = True
                break

        # Apply friction (reduced on ice)
        if self.isOnGround:
            if self.onIcePlatform:
                # Ice: very low friction (0.95 = keeps 95% of velocity)
                self.velocityX *= 0.95
                # Cap velocity on ice to prevent infinite acceleration
                max_ice_velocity = 15
                if abs(self.velocityX) > max_ice_velocity:
                    self.velocityX = math.copysign(max_ice_velocity, self.velocityX)
            else:
                # Normal: high friction (0.7 = keeps 70% of velocity)
                self.velocityX *= 0.7

        # Track distance
        self.distance = max(self.distance, self.x)

        # Die if fall off map
        if self.y > 1200:
            self.die()

    def checkPlatformCollision(self, platform):
        """
        Check if player is falling and landing on top of platform.

        Args:
            platform: Platform dictionary with x, y, width, height

        Returns:
            bool: True if collision detected
        """
        # Check if player is falling
        if self.velocityY >= 0:
            playerBottom = self.y + self.height
            platformTop = platform['y']

            # Check if player's feet are near platform top
            if (playerBottom >= platformTop and
                playerBottom <= platformTop + self.velocityY + self.gravity + 5 and
                self.x + self.width > platform['x'] and
                self.x < platform['x'] + platform['width']):
                return True

        return False

    def moveLeft(self):
        """Move player left (or faster if sprinting)"""
        speed = self.speed * 1.5 if self.isSprinting else self.speed
        if self.onIcePlatform:
            # On ice: add to velocity (acceleration-based for sliding)
            self.velocityX -= speed * 0.3  # 30% acceleration on ice
        else:
            # Normal: direct control
            self.velocityX = -speed

    def moveRight(self):
        """Move player right (or faster if sprinting)"""
        speed = self.speed * 1.5 if self.isSprinting else self.speed
        if self.onIcePlatform:
            # On ice: add to velocity (acceleration-based for sliding)
            self.velocityX += speed * 0.3  # 30% acceleration on ice
        else:
            # Normal: direct control
            self.velocityX = speed

    def jump(self):
        """Jump if on ground and not already jumping"""
        if self.isOnGround and self.velocityY >= 0:
            self.velocityY = -self.jumpPower
            self.isOnGround = False

    def sprint(self, active):
        """Set sprint state"""
        self.isSprinting = active

    def duck(self, active):
        """Set duck state (could reduce height when ducking)"""
        self.isDucking = active

    def stopMovement(self):
        """Stop horizontal movement"""
        self.velocityX = 0

    def die(self):
        """Kill the player"""
        self.isAlive = False
        self.velocityX = 0
        self.velocityY = 0

    def reset(self, x, y):
        """Reset player to starting position"""
        self.x = x
        self.y = y
        self.velocityX = 0
        self.velocityY = 0
        self.isAlive = True
        self.isOnGround = False
        self.distance = 0
        self.score = 0

    def collectCoin(self, coin):
        """
        Check if player collected a coin.

        Args:
            coin: Coin dictionary with x, y, radius, collected

        Returns:
            bool: True if coin was collected
        """
        if not coin.get('collected', False):
            dx = self.x + self.width / 2 - coin['x']
            dy = self.y + self.height / 2 - coin['y']
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < self.width / 2 + coin['radius']:
                coin['collected'] = True
                self.score += 10
                return True

        return False

    def checkEnemyCollision(self, enemy):
        """
        Check AABB collision with enemy.

        Args:
            enemy: Enemy dictionary with x, y, width, height

        Returns:
            bool: True if collision detected
        """
        return (
            self.x < enemy['x'] + enemy['width'] and
            self.x + self.width > enemy['x'] and
            self.y < enemy['y'] + enemy['height'] and
            self.y + self.height > enemy['y']
        )

    def checkGoalReached(self, goal):
        """
        Check if player reached the goal.

        Args:
            goal: Goal dictionary with x, y, width, height

        Returns:
            bool: True if goal reached
        """
        return (
            self.x < goal['x'] + goal['width'] and
            self.x + self.width > goal['x'] and
            self.y < goal['y'] + goal['height'] and
            self.y + self.height > goal['y']
        )
