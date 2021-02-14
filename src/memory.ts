import * as tf from '@tensorflow/tfjs'
import { sampleSize } from 'lodash'

type Sample = [tf.Tensor2D, number, number, tf.Tensor2D | null]

export class Memory {
  public maxMemory: number
  public samples: Array<Sample>

  /**
   * @param {number} maxMemory
   */
  constructor(maxMemory: number) {
    this.maxMemory = maxMemory
    this.samples = new Array()
  }

  /**
   * @param {Array} sample
   */
  public addSample(sample: Sample) {
    this.samples.push(sample)
    if (this.samples.length > this.maxMemory) {
      let [state, _action, _reward, nextState] = this.samples.shift() ?? []
      state?.dispose()
      nextState?.dispose()
    }
  }

  /**
   * @param {number} nSamples
   * @returns {Array} Randomly selected samples
   */
  public sample(nSamples: number) {
    return sampleSize(this.samples, nSamples)
  }
}
