const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')

function sexToNumber(sex) {
  switch (sex) {
    case 'F': return 0
    case 'M': return 1
    default: return 0.5
  }
}

function getCsvSize(filename) {
  const lines = fs.readFileSync(filename, 'utf-8').split("\n")
  return {
    rows: lines.length - 1,
    columns: lines[0].split(",").length
  }
}

function prepareData(filename) {
  const options = { hasHeader: true, columnConfigs: { rings: { isLabel: true } } }

  return tf.data.csv(`file://${filename}`, options).map(row => ({
    xs: Object.values(row.xs).map((x, i) => i == 0 ? sexToNumber(x) : x),
    ys: [row.ys.rings]
  }))
}

function createModel(inputShape, activation = 'sigmoid', lr = 0.01) {
  const model = tf.sequential()
  model.add(tf.layers.dense({ inputShape, activation, units: inputShape[0] * 2 }))
  model.add(tf.layers.dense({ units: 1 }))
  model.compile({ optimizer: tf.train.sgd(lr), loss: 'meanSquaredError' })
  return model
}

async function train({ model, data, numRows, batchSize = 100, epochs = 200, trainRatio = .75 }) {
  const trainLength = Math.floor(numRows + trainRatio)
  const trainBatches = Math.floor(trainLength / batchSize)
  const shuffled = data.shuffle(100).batch(batchSize)
  const trainData = shuffled.take(trainBatches)
  const testData = shuffled.skip(trainBatches)
  await model.fitDataset(trainData, { epochs, validationData: testData })
}

const tests = [
  [1, 0.365, 0.295, 0.08, 0.2555, 0.097, 0.043, 0.1],
  [1, 0.45, 0.32, 0.1, 0.381, 0.1705, 0.075, 0.115],
  [1, 0.355, 0.28, 0.095, 0.2455, 0.0955, 0.062, 0.075],
  [0, 0.38, 0.275, 0.1, 0.2255, 0.08, 0.049, 0.085]
]

async function main(csvName) {
  const data = prepareData(csvName);
  const size = getCsvSize(csvName)
  const model = createModel([size.columns - 1])
  await train({ model, data, numRows: size.rows })
  for (let i = 0; i < tests.length; i++) {
    const test = tests[i]
    const output = model.predict(tf.tensor2d([test]))
    console.log(output.dataSync());

  }
}

const csvName = './data/abalone.csv'

main(csvName)
