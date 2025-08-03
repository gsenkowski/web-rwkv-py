use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
    sync::Arc,
};

use std::time::Instant;

use anyhow::{bail, Result};
use derivative::Derivative;
use half::f16;
use memmap2::Mmap;
use pyo3::{exceptions::PyValueError, prelude::*};
use safetensors::SafeTensors;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{Token},
        infer::rnn::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{
            ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant,
            State as ModelState, Bundle
        },
        v4, v5, v6, v7, TokioRuntime,
    },
    tensor::{
        kind::ReadWrite, DeepClone, TensorCpu, TensorGpu, TensorInit, TensorInto, TensorShape,
    },
    wgpu,
};


pub mod info;

fn err(err: impl ToString) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// A model with runtime.
#[pyclass]
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct Model {
    tokio: Arc<tokio::runtime::Runtime>,
    info: ModelInfo,
    context: Context,
    runtime: TokioRuntime<Rnn>,
    #[derivative(Debug = "ignore")]
    state: Arc<dyn ModelState + Send + Sync>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum State {
    Cpu { state: StateCpu },
    Gpu { state: StateGpu },
}

#[pymethods]
impl State {
    pub fn deep_clone(&self) -> Self {
        match self.clone() {
            State::Cpu { state } => State::Cpu { state },
            State::Gpu { state } => {
                let state = StateGpu(state.0.deep_clone());
                State::Gpu { state }
            }
        }
    }

    pub fn device(&self) -> StateDevice {
        match self {
            State::Cpu { .. } => StateDevice::Cpu,
            State::Gpu { .. } => StateDevice::Gpu,
        }
    }

    pub fn to(&self, device: StateDevice) -> Self {
        match (self.clone(), device) {
            (Self::Cpu { state }, StateDevice::Gpu) => {
                let StateCpu(tensor, context) = state;
                let state = StateGpu(tensor.to(&context));
                Self::Gpu { state }
            }
            (Self::Gpu { state }, StateDevice::Cpu) => {
                let context = state.0.context.clone();
                let tensor = state.0.back_in_place();
                let state = StateCpu(tensor, context);
                Self::Cpu { state }
            }
            (state, _) => state,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct StateCpu(TensorCpu<f32>, Context);

#[pyclass]
#[derive(Debug, Clone)]
pub struct StateGpu(TensorGpu<f32, ReadWrite>);

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateDevice {
    Cpu,
    Gpu,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = web_rwkv::wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

async fn load_runtime(
    path: PathBuf,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
) -> Result<(
    Context,
    ModelInfo,
    TokioRuntime<Rnn>,
    Arc<dyn ModelState + Send + Sync>,
)> {
    let file = File::open(path)?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;

    let context = create_context(&info).await?;
    let quant = (0..quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
        .chain((0..quant_sf4).map(|layer| (layer, Quant::SF4)))
        .collect();

    let builder = ModelBuilder::new(&context, model).quant(quant);

    match info.version {
        ModelVersion::V4 => {
            let model = builder.build_v4().await?;
            let bundle = v4::Bundle::<f16>::new(model, 1);
            let state = Arc::new(bundle.state());
            let runtime = TokioRuntime::new(bundle).await;
            Ok((context, info, runtime, state))
        }
        ModelVersion::V5 => {
            let model = builder.build_v5().await?;
            let bundle = v5::Bundle::<f16>::new(model, 1);
            let state = Arc::new(bundle.state());
            let runtime = TokioRuntime::new(bundle).await;
            Ok((context, info, runtime, state))
        }
        ModelVersion::V6 => {
            let model = builder.build_v6().await?;
            let bundle = v6::Bundle::<f16>::new(model, 1);
            let state = Arc::new(bundle.state());
            let runtime = TokioRuntime::new(bundle).await;
            Ok((context, info, runtime, state))
        }
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            let state = Arc::new(bundle.state());
            let runtime = TokioRuntime::new(bundle).await;
            Ok((context, info, runtime, state))
        }
    }
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (path, quant=0, quant_nf4=0, quant_sf4=0))]
    pub fn new(path: PathBuf, quant: usize, quant_nf4: usize, quant_sf4: usize) -> PyResult<Self> {
        let tokio = Arc::new(tokio::runtime::Runtime::new()?);
        let (context, info, runtime, state) = tokio
            .block_on(load_runtime(path, quant, quant_nf4, quant_sf4))
            .map_err(err)?;
        Ok(Self {
            tokio,
            context,
            info,
            runtime,
            state,
        })
    }

    pub fn info(&self) -> info::ModelInfo {
        self.info.clone().into()
    }

    pub fn init_state(&self) -> State {
        let state = StateCpu(self.state.init(), self.context.clone());
        State::Cpu { state }
    }

    #[pyo3(signature = (filename, device=StateDevice::Cpu))]
    pub fn save_state_to_file(&self, filename: PathBuf, device: StateDevice) -> PyResult<()> {
        // Retrieve the state on the requested device and convert to CPU for saving.
        let state = self.back_state(device)?;
        let state_cpu = state.to(StateDevice::Cpu);
        if let State::Cpu { state: StateCpu(tensor, _context) } = state_cpu {
            // Collect dimensions using the public iterator.
            let dims_vec: Vec<usize> = tensor.shape().iter().collect();
            if dims_vec.len() != 4 {
                return Err(err("Expected tensor shape of length 4"));
            }
            let dims: [usize; 4] = dims_vec[..]
                .try_into()
                .map_err(|_| err("Failed to convert shape slice to array"))?;
            let data = tensor.to_vec();

            // Open file and wrap with BufWriter.
            let file = File::create(filename).map_err(err)?;
            let mut writer = BufWriter::new(file);

            // Write the number of dimensions.
            let num_dims = dims.len() as u64;
            writer.write_all(&num_dims.to_le_bytes()).map_err(err)?;
            // Write each dimension.
            for &dim in &dims {
                writer.write_all(&(dim as u64).to_le_bytes()).map_err(err)?;
            }
            // Write the number of data elements.
            writer.write_all(&(data.len() as u64).to_le_bytes()).map_err(err)?;

            // Safely convert the f32 data to a u8 vector without unsafe.
            let data_bytes: Vec<u8> = data.iter()
                .flat_map(|val| val.to_le_bytes())
                .collect();
            writer.write_all(&data_bytes).map_err(err)?;
            writer.flush().map_err(err)?;
            Ok(())
        } else {
            Err(err("Unexpected state variant"))
        }
    }

    #[pyo3(signature = (filename, device=StateDevice::Cpu))]
    pub fn load_state_from_file(&self, filename: PathBuf, device: StateDevice) -> PyResult<()> {
        let file = File::open(filename).map_err(err)?;
        let mut reader = BufReader::new(file);

        // Read the number of dimensions.
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8).map_err(err)?;
        let num_dims = u64::from_le_bytes(buf8) as usize;

        // Read each dimension.
        let mut dims_vec = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            reader.read_exact(&mut buf8).map_err(err)?;
            dims_vec.push(u64::from_le_bytes(buf8) as usize);
        }
        if dims_vec.len() != 4 {
            return Err(err("Expected tensor shape of length 4"));
        }
        let dims: [usize; 4] = [dims_vec[0], dims_vec[1], dims_vec[2], dims_vec[3]];

        // Read the number of data elements.
        reader.read_exact(&mut buf8).map_err(err)?;
        let data_len = u64::from_le_bytes(buf8) as usize;

        // Allocate a buffer for the tensor data.
        let num_bytes = data_len * std::mem::size_of::<f32>();
        let mut data_bytes = vec![0u8; num_bytes];
        reader.read_exact(&mut data_bytes).map_err(err)?;

        // Convert the byte buffer to a Vec<f32> safely.
        let data: Vec<f32> = data_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk
                    .try_into()
                    .expect("chunks_exact always returns a slice of length 4");
                f32::from_le_bytes(arr)
            })
            .collect();

        // Reconstruct the tensor from the shape and data.
        let tensor = TensorCpu::from_data(dims, data).map_err(err)?;
        let state_cpu = StateCpu(tensor, self.context.clone());
        let state = State::Cpu { state: state_cpu }.to(device);
        self.load_state(&state)
    }
    
    pub fn clear_state(&self) {
        let _ = self.load_state(&self.init_state());
    }

    pub fn load_state(&self, state: &State) -> PyResult<()> {
        match state.clone() {
            State::Cpu { state } => self.state.load(state.0, 0),
            State::Gpu { state } => self.state.write(state.0, 0),
        }
        .map_err(err)
    }

    #[pyo3(signature = (device=StateDevice::Cpu))]
    pub fn back_state(&self, device: StateDevice) -> PyResult<State> {
        match device {
            StateDevice::Cpu => {
                let tensor = self.tokio.block_on(self.state.back(0)).map_err(err)?;
                let state = StateCpu(tensor, self.context.clone());
                Ok(State::Cpu { state })
            }
            StateDevice::Gpu => {
                let tensor = self.state.read(0).map_err(err)?;
                let state = StateGpu(tensor);
                Ok(State::Gpu { state })
            }
        }
    }

    #[pyo3(signature = (tokens, token_chunk_size=128))]
    pub fn run(&self, tokens: Vec<u16>, token_chunk_size: usize) -> PyResult<Vec<f32>> {
        let model = self.clone();
        let option = RnnOption::Last;
        let output = self
            .tokio
            .block_on(model.run_internal(tokens, option, token_chunk_size))
            .map_err(err)?
            .to_vec();
        Ok(output)
    }

    #[pyo3(signature = (tokens, token_chunk_size=128))]
    pub fn run_full(&self, tokens: Vec<u16>, token_chunk_size: usize) -> PyResult<Vec<f32>> {
        let model = self.clone();
        let option = RnnOption::Full;
        let output = self
            .tokio
            .block_on(model.run_internal(tokens, option, token_chunk_size))
            .map_err(err)?
            .to_vec();
        Ok(output)
    }
}

impl Model {
    async fn run_internal(
        &self,
        tokens: Vec<u16>,
        option: RnnOption,
        token_chunk_size: usize,
    ) -> Result<TensorCpu<f32>> {
        if tokens.is_empty() {
            bail!("input tokens cannot be empty")
        }
        
        let tokens_token: Vec<Token> = tokens.into_iter().map(Token::from).collect();
        let mut inference = Some(RnnInput::new(
            vec![RnnInputBatch::new(tokens_token, option)],
            token_chunk_size,
        ));
        let mut data = vec![];
        let mut num_token = 0;
        loop {
            let input = inference.take().unwrap();
            if input.batches[0].tokens.is_empty() {
                break;
            }

            let (input, output) = match self.runtime.infer(input).await {
                Ok(output) => output,
                Err(err) => {
                    bail!(err);
                }
            };

            num_token += output[0].0.shape()[1];
            let mut output = output[0].clone().to_vec();
            data.append(&mut output);
            inference.replace(input);
        }

        let num_vocab = self.info.num_vocab;
        let tensor = TensorCpu::from_data([num_vocab, num_token, 1, 1], data)?;
        Ok(tensor)
    }
}

fn load_tokenizer(path: PathBuf) -> Result<web_rwkv::tokenizer::Tokenizer> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(web_rwkv::tokenizer::Tokenizer::new(&contents)?)
}

#[pyclass]
pub struct Tokenizer(web_rwkv::tokenizer::Tokenizer);

#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self(load_tokenizer(path).map_err(err)?))
    }

    pub fn encode(&self, text: &str) -> PyResult<Vec<u16>> {
        self.0.encode(text.as_bytes()).map_err(err)
    }

    pub fn decode(&self, tokens: Vec<u16>) -> PyResult<Vec<u8>> {
        self.0.decode(&tokens).map_err(err)
    }
}

#[pymodule]
fn web_rwkv_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<State>()?;
    module.add_class::<StateDevice>()?;
    module.add_class::<Tokenizer>()?;
    module.add_class::<info::ModelInfo>()?;
    module.add_class::<info::ModelVersion>()?;

    Ok(())
}
