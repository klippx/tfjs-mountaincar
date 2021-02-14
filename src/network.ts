import * as tf from '@tensorflow/tfjs'

import { Model } from './model'
import { Orchestrator } from './orchestrator'
import { onGameEnd } from './ui'
import { Memory } from './memory'
import { MountainCar } from './mountainCar'

/**
 * Policy network for controlling the cart-pole system.
 *
 * The role of the policy network is to select an action based on the observed
 * state of the system. In this case, the action is the leftward or rightward
 * force and the observed system state is a four-dimensional vector, consisting
 * of cart position, cart velocity, pole angle and pole angular velocity.
 *
 */
class PolicyNetwork {
  public memory: Memory
  public model: Model

  /**
   * Constructor of PolicyNetwork.
   *
   * @param {number | number[] | tf.LayersModel} hiddenLayerSizes
   *   Can be any of the following
   *   - Size of the hidden layer, as a single number (for a single hidden
   *     layer)
   *   - An Array of numbers (for any number of hidden layers).
   *   - An instance of tf.LayersModel.
   */
  constructor(hiddenLayerSizesOrModel: number | number[] | tf.LayersModel) {
    this.memory = new Memory(500)
    this.model = new Model(hiddenLayerSizesOrModel, 2, 3, 100)
  }

  /**
   * Train the policy network's model.
   *
   * @param {CartPole} cartPoleSystem The cart-pole system object to use during
   *   training.
   * @param {tf.train.Optimizer} optimizer An instance of TensorFlow.js
   *   Optimizer to use for training.
   * @param {number} discountRate Reward discounting rate: a number between 0
   *   and 1.
   * @param {number} numGames Number of game to play for each model parameter
   *   update.
   * @param {number} maxStepsPerGame Maximum number of steps to perform during
   *   a game. If this number is reached, the game will end immediately.
   * @returns {number[]} The number of steps completed in the `numGames` games
   *   in this round of training.
   */
  async train(
    cartPoleSystem: MountainCar,
    discountRate: number,
    numGames: number,
    maxStepsPerGame: number
  ) {
    const maxPositionStore: Array<number> = new Array()
    onGameEnd(0, numGames)
    for (let i = 0; i < numGames; ++i) {
      // Randomly initialize the state of the cart-pole system at the beginning
      // of every game.
      const orchestrator = new Orchestrator(
        cartPoleSystem,
        this.model,
        this.memory,
        discountRate,
        maxStepsPerGame
      )
      await orchestrator.run()
      maxPositionStore.push(
        orchestrator.maxPositionStore[orchestrator.maxPositionStore.length - 1]
      )
      onGameEnd(i + 1, numGames)
    }
    return Math.max(...maxPositionStore)
  }
}

// The IndexedDB path where the model of the policy network will be saved.
const MODEL_SAVE_PATH_ = 'indexeddb://mountain-car-v0'

/**
 * A subclass of PolicyNetwork that supports saving and loading.
 */
export class SaveablePolicyNetwork extends PolicyNetwork {
  /**
   * Constructor of SaveablePolicyNetwork
   *
   * @param {number | tf.LayersModel} hiddenLayerSizesOrModel
   */
  constructor(hiddenLayerSizesOrModel: number | number[] | tf.LayersModel) {
    super(hiddenLayerSizesOrModel)
  }

  /**
   * Save the model to IndexedDB.
   */
  async saveModel() {
    return await this.model.network.save(MODEL_SAVE_PATH_)
  }

  /**
   * Load the model fom IndexedDB.
   *
   * @returns {SaveablePolicyNetwork} The instance of loaded
   *   `SaveablePolicyNetwork`.
   * @throws {Error} If no model can be found in IndexedDB.
   */
  static async loadModel() {
    const modelsInfo = await tf.io.listModels()
    if (MODEL_SAVE_PATH_ in modelsInfo) {
      console.log(`Loading existing model...`)
      const model = await tf.loadLayersModel(MODEL_SAVE_PATH_)
      console.log(`Loaded model from ${MODEL_SAVE_PATH_}`)
      return new SaveablePolicyNetwork(model)
    } else {
      throw new Error(`Cannot find model at ${MODEL_SAVE_PATH_}.`)
    }
  }

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  static async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels()
    return modelsInfo[MODEL_SAVE_PATH_]
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_)
  }

  /**
   * Get the sizes of the hidden layers.
   *
   * @returns {number | number[]} If the model has only one hidden layer,
   *   return the size of the layer as a single number. If the model has
   *   multiple hidden layers, return the sizes as an Array of numbers.
   */
  hiddenLayerSizes(): number | number[] {
    const sizes = []
    for (let i = 0; i < this.model.network.layers.length - 1; ++i) {
      // FIXME: as any
      const x = this.model.network.layers[i] as any
      console.log(Object.keys(x), { units: x.units })
      sizes.push(x.units)
    }
    return sizes.length === 1 ? sizes[0] : sizes
  }
}
