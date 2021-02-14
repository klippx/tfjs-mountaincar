import * as tf from '@tensorflow/tfjs'

export class Model {
  public numStates: number
  public numActions: number
  public batchSize: number
  public network: tf.Sequential | tf.LayersModel

  /**
   * @param {number} numStates
   * @param {number} numActions
   * @param {number} batchSize
   */

  constructor(
    hiddenLayerSizesOrModel: tf.LayersModel | number[] | number,
    numStates: number,
    numActions: number,
    batchSize: number
  ) {
    this.numStates = numStates
    this.numActions = numActions
    this.batchSize = batchSize

    if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
      this.network = hiddenLayerSizesOrModel
      this.network.summary()
      this.network.compile({ optimizer: 'adam', loss: 'meanSquaredError' })
    } else {
      let hiddenLayerSizes: Array<number> = []
      if (Array.isArray(hiddenLayerSizesOrModel)) {
        hiddenLayerSizes = hiddenLayerSizesOrModel
      } else {
        hiddenLayerSizes = [hiddenLayerSizesOrModel]
      }
      const network = tf.sequential()
      hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
        network.add(
          tf.layers.dense({
            units: hiddenLayerSize,
            activation: 'relu',
            // `inputShape` is required only for the first layer.
            inputShape: i === 0 ? [this.numStates] : undefined,
          })
        )
      })
      network.add(tf.layers.dense({ units: this.numActions }))

      network.summary()
      network.compile({ optimizer: 'adam', loss: 'meanSquaredError' })
      this.network = network
    }
  }

  /**
   * @param {tf.Tensor | tf.Tensor[]} states
   * @returns {tf.Tensor | tf.Tensor[]} The predictions of the best actions
   */
  predict(states: tf.Tensor | tf.Tensor[]) {
    return tf.tidy(() => this.network.predict(states))
  }

  /**
   * @param {tf.Tensor[]} xBatch
   * @param {tf.Tensor[]} yBatch
   */
  async train(xBatch: tf.Tensor2D, yBatch: tf.Tensor2D) {
    await this.network.fit(xBatch, yBatch)
  }

  /**
   * @param {tf.Tensor} state
   * @returns {number} The action chosen by the model (-1 | 0 | 1)
   */
  chooseAction(state: tf.Tensor, eps: number): number {
    if (Math.random() < eps) {
      return Math.floor(Math.random() * this.numActions) - 1
    } else {
      return tf.tidy(() => {
        // FIXME: as tf.Tensor<tf.Rank>
        const logits = this.network.predict(state) as tf.Tensor<tf.Rank>
        const sigmoid = tf.sigmoid(logits)
        const probs = tf.div(sigmoid, tf.sum(sigmoid))
        // FIXME: as any
        return tf.multinomial(probs as any, 1).dataSync()[0] - 1
      })
    }
  }
}
