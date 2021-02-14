import * as tf from '@tensorflow/tfjs'

/**
 * Mountain car system simulator.
 *
 * There are two state variables in this system:
 *
 *   - position: The x-coordinate of location of the car.
 *   - velocity: The velocity of the car.
 *
 * The system is controlled through three distinct actions:
 *
 *   - leftward acceleration.
 *   - rightward acceleration
 *   - no acceleration
 */
export class MountainCar {
  // constructor
  public minPosition: number
  public maxPosition: number
  private maxSpeed: number
  private goalPosition: number
  private goalVelocity: number
  private gravity: number
  public carWidth: number
  public carHeight: number
  private force: number

  // setRandomState
  public position: number
  private velocity: number

  /**
   * Constructor of MountainCar.
   */
  constructor() {
    // Constants that characterize the system.

    this.minPosition = -1.2
    this.maxPosition = 0.6
    this.maxSpeed = 0.07
    this.goalPosition = 0.5
    this.goalVelocity = 0
    this.gravity = 0.0025
    this.carWidth = 0.2
    this.carHeight = 0.1
    this.force = 0.0013

    // setRandomState()
    this.position = Math.random() / 5 - 0.6
    this.velocity = 0
  }

  /**
   * Set the state of the mountain car system randomly.
   */
  public setRandomState() {
    // The state variables of the mountain car system.
    // Car position
    this.position = Math.random() / 5 - 0.6
    // Car velocity.
    this.velocity = 0
  }

  /**
   * Get current state as a tf.Tensor of shape [1, 2].
   */
  public getStateTensor() {
    return tf.tensor2d([[this.position, this.velocity]])
  }

  /**
   * Update the mountain car system using an action.
   * @param {number} action Only the sign of `action` matters.
   *   Action is an integer, in [-1, 0, 1]
   *   A value of 1 leads to a rightward force of a fixed magnitude.
   *   A value of -1 leads to a leftward force of the same fixed magnitude.
   *   A value of 0 leads to no force applied.
   * @returns {boolean} Whether the simulation is done.
   */
  public update(action: number): boolean {
    this.velocity +=
      action * this.force - Math.cos(3 * this.position) * this.gravity
    this.velocity = Math.min(
      Math.max(this.velocity, -this.maxSpeed),
      this.maxSpeed
    )

    this.position += this.velocity
    this.position = Math.min(
      Math.max(this.position, this.minPosition),
      this.maxPosition
    )

    if (this.position == this.minPosition && this.velocity < 0) {
      this.velocity = 0
    }

    return this.isDone()
  }

  /**
   * Determine whether this simulation is done.
   *
   * A simulation is done when `position` reaches `goalPosition`
   * and `velocity` is greater than zero.
   *
   * @returns {boolean} Whether the simulation is done.
   */
  public isDone(): boolean {
    return (
      this.position >= this.goalPosition && this.velocity >= this.goalVelocity
    )
  }
}
