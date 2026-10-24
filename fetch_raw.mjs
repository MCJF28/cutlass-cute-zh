import fs from 'node:fs';
import path from 'node:path';
import https from 'node:https';

const BASE_URL_CUTE = 'https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/cute/';
const BASE_URL_CPP = 'https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/';

const CUTE_FILES = [
  '00_quickstart.md',
  '01_layout.md',
  '02_layout_algebra.md',
  '03_tensor.md',
  '04_algorithms.md',
  '0t_mma_atom.md',
  '0x_gemm_tutorial.md',
  '0y_predication.md',
  '0z_tma_tensors.md'
];

const CPP_FILES = [
  'cutlass_3x_design.md',
  'gemm_api_3x.md',
  'pipeline.md',
  'efficient_gemm.md',
  'blackwell_functionality.md'
];

const targetCute = path.resolve('cutlass-cute-zh', 'raw', 'cute');
const targetCpp = path.resolve('cutlass-cute-zh', 'raw', 'cpp');
fs.mkdirSync(targetCute, { recursive: true });
fs.mkdirSync(targetCpp, { recursive: true });

function download(url, dest) {
  return new Promise((resolve) => {
    https.get(url, (res) => {
      if (res.statusCode === 200) {
        const fileStream = fs.createWriteStream(dest);
        res.pipe(fileStream);
        fileStream.on('finish', () => {
          fileStream.close();
          console.log(`Saved: ${dest}`);
          resolve(true);
        });
      } else {
        console.warn(`Failed ${url}: HTTP ${res.statusCode}`);
        resolve(false);
      }
    }).on('error', (err) => {
      console.error(`Error ${url}: ${err.message}`);
      resolve(false);
    });
  });
}

async function run() {
  console.log('Downloading CuTe docs...');
  for (const f of CUTE_FILES) {
    await download(BASE_URL_CUTE + f, path.join(targetCute, f));
  }
  console.log('Downloading CUTLASS 3.x / Architecture docs...');
  for (const f of CPP_FILES) {
    await download(BASE_URL_CPP + f, path.join(targetCpp, f));
  }
  console.log('Finished downloading all raw docs!');
}

run();
