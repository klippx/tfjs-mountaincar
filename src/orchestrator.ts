import { MountainCar } from './mountainCar'
import { Model } from './model'
import { Memory } from './memory'
import { maybeRenderDuringTraining, setUpUI } from './ui'

import * as tf from '@tensorflow/tfjs'

const MIN_EPSILON = 0.01
const MAX_EPSILON = 0.2
const LAMBDA = 0.01

export class Orchestrator {
  public mountainCar: MountainCar
  private model: Model
  private memory: Memory
  private eps: number
  private steps: number
  private maxStepsPerGame: number
  private discountRate: number
  private rewardStore: Array<number>
  public maxPositionStore: Array<number>
  private stopRequested: () => boolean

  /**
   * @param {MountainCar} mountainCar
   * @param {Model} model
   * @param {Memory} memory
   * @param {number} discountRate
   * @param {number} maxStepsPerGame
   */
  constructor(
    mountainCar: MountainCar,
    model: Model,
    memory: Memory,
    discountRate: number,
    maxStepsPerGame: number,
    stopRequested: () => boolean
  ) {
    // The main components of the environment
    this.mountainCar = mountainCar
    this.model = model
    this.memory = memory

    // The exploration parameter
    this.eps = MAX_EPSILON

    // Keep tracking of the elapsed steps
    this.steps = 0
    this.maxStepsPerGame = maxStepsPerGame
    this.discountRate = discountRate

    // Initialization of the rewards and max positions containers
    this.rewardStore = new Array()
    this.maxPositionStore = new Array()
    this.stopRequested = stopRequested
  }

  /**
   * @param {number} position
   * @returns {number} Reward corresponding to the position
   */
  computeReward(position: number): number {
    let reward = 0
    if (position >= 0) {
      reward = 5
    } else if (position >= 0.1) {
      reward = 10
    } else if (position >= 0.25) {
      reward = 20
    } else if (position >= 0.5) {
      reward = 100
    }
    return reward
  }

  async run() {
    this.mountainCar.setRandomState()
    let state = this.mountainCar.getStateTensor()
    let totalReward = 0
    let maxPosition = -100
    let step = 0
    while (step < this.maxStepsPerGame && this.stopRequested() === false) {
      // Rendering in the browser
      await maybeRenderDuringTraining(this.mountainCar)

      // Interaction with the environment
      const action = this.model.chooseAction(state, this.eps)
      const done = this.mountainCar.update(action)
      const reward = this.computeReward(this.mountainCar.position)

      let nextState: tf.Tensor2D | null = this.mountainCar.getStateTensor()

      // Keep the car on max position if reached
      if (this.mountainCar.position > maxPosition) {
        maxPosition = this.mountainCar.position
      }

      if (done) {
        nextState = null
      }

      this.memory.addSample([state, action, reward, nextState])

      this.steps += 1

      // Exponentially decay the exploration parameter
      this.eps =
        MIN_EPSILON +
        (MAX_EPSILON - MIN_EPSILON) * Math.exp(-LAMBDA * this.steps)

      if (nextState !== null) {
        state = nextState
      }
      totalReward += reward
      step += 1

      // Keep track of the max position reached and store the total reward
      if (done || step == this.maxStepsPerGame) {
        this.rewardStore.push(totalReward)
        this.maxPositionStore.push(maxPosition)
        break
      }
    }
    await this.replay()
  }

  async replay() {
    // Sample from memory
    const batch = this.memory.sample(this.model.batchSize)
    const states = batch.map(([state, , ,]) => state)
    const nextStates = batch.map(([, , , nextState]) =>
      nextState ? nextState : tf.zeros([this.model.numStates])
    )

    // Predict the values of each action at each state
    const qsa = states.map((state) => {
      // FIXME: as tf.Tensor<tf.Rank>
      return this.model.predict(state) as tf.Tensor<tf.Rank>
    })

    // Predict the values of each action at each next state
    const qsad = nextStates.map((nextState) => {
      // FIXME: as tf.Tensor<tf.Rank>
      return this.model.predict(nextState) as tf.Tensor<tf.Rank>
    })

    // FIXME: as any
    const x = new Array()
    const y = new Array()

    // Update the states rewards with the discounted next states rewards
    batch.forEach(([state, action, reward, nextState], index) => {
      // FIXME: as any
      const currentQ = qsa[index] as any
      const nextQ = qsad[index] as any

      currentQ[action] = nextState
        ? reward + this.discountRate * nextQ.max().dataSync()
        : reward

      x.push(state.dataSync())
      y.push(currentQ.dataSync())
    })

    // Clean unused tensors
    qsa.forEach((state) => state.dispose())
    qsad.forEach((state) => state.dispose())

    // Reshape the batches to be fed to the network
    const tx = tf.tensor2d(x, [x.length, this.model.numStates])
    const ty = tf.tensor2d(y, [y.length, this.model.numActions])

    // Learn the Q(s, a) values given associated discounted rewards
    await this.model.train(tx, ty)

    tx.dispose()
    ty.dispose()
  }
}
