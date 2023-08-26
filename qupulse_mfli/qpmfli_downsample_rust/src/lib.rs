use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, IntoPyArray};
use ndarray::{Array1, ArrayView1};
use packed_simd::f32x8; // Assuming 256-bit wide SIMD instructions are available. 
use rayon::prelude::*;
#[macro_use(s)]
extern crate ndarray;

fn find_start_index(sorted_time_data: ArrayView1<f32>, target: f32) -> usize {
    sorted_time_data
        .as_slice().unwrap()  // Convert ArrayView1 to a slice
        .binary_search_by(|&probe| probe.partial_cmp(&target).unwrap())
        .unwrap_or_else(|e| e)
}

fn extract_data_segment(
    applicable_data: ArrayView1<f32>,
    time_data: ArrayView1<f32>,
    begins: &Array1<f32>,
    lengths: &Array1<f32>,
    time_of_trigger: f32,
) -> Vec<f32> {
    begins
        .as_slice().unwrap()
        .par_iter()
        .zip(lengths.as_slice().unwrap().par_iter())
        .map(|(&b, &l)| {
            let lower_bound = time_of_trigger + b;
            let upper_bound = time_of_trigger + b + l;

            let start_index = find_start_index(time_data, lower_bound);

            let simd_chunk = f32x8::splat(upper_bound);
            let mut sum_simd = f32x8::splat(0.0);
            let mut count = 0;

            let applicable_data_slice = &applicable_data.as_slice().unwrap()[start_index..];
            let time_data_slice = &time_data.as_slice().unwrap()[start_index..];

            let mut sum: f32 = 0.0;  // Declare sum as mutable right from the beginning

			for (data_chunk, time_chunk) in applicable_data_slice.chunks_exact(8).zip(time_data_slice.chunks_exact(8)) {
				let data_chunk_simd = f32x8::from_slice_unaligned(data_chunk);
				let time_chunk_simd = f32x8::from_slice_unaligned(time_chunk);

				let lower_bound_simd = f32x8::splat(lower_bound);
				let mask = time_chunk_simd.ge(lower_bound_simd) & time_chunk_simd.lt(simd_chunk);

				for i in 0..8 {
					if mask.extract(i) {
						sum += data_chunk_simd.extract(i);
						count += 1;
					}
				}
			}

            //let mut sum: f32 = sum_simd.sum();
			
			let remaining_elements = applicable_data_slice.len() % 8;
			let offset = applicable_data_slice.len() - remaining_elements;
			for i in 0..remaining_elements {
				let time_value = time_data_slice[offset + i];
				if time_value >= lower_bound && time_value < upper_bound {
					sum += applicable_data_slice[offset + i];
					count += 1;
				}
			}
			
            if count > 0 {
                sum / count as f32
            } else {
                std::f32::NAN
            }
        })
        .collect()
}

fn extract_data(
    applicable_data: &Array1<f32>,
    time_data: &Array1<f32>,
    begins: &Array1<f32>,
    lengths: &Array1<f32>,
    time_of_trigger: f32,
) -> Vec<f32> {
    let segment_size = applicable_data.len() / 8; // Assuming 8 segments

    (0..8)
        .into_par_iter()
        .flat_map(|i| {
            extract_data_segment(
                applicable_data.slice(s![i * segment_size..(i + 1) * segment_size]),
                time_data.slice(s![i * segment_size..(i + 1) * segment_size]),
                begins,
                lengths,
                time_of_trigger,
            )
        })
        .collect()
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn py_extract_data(
    py: Python,
    applicable_data: &PyArray1<f32>,
    time_data: &PyArray1<f32>,
    begins: &PyArray1<f32>,
    lengths: &PyArray1<f32>,
    time_of_trigger: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let applicable_data = unsafe { applicable_data.as_array().to_owned() };
    let time_data = unsafe { time_data.as_array().to_owned() };
    let begins = unsafe { begins.as_array().to_owned() };
    let lengths = unsafe { lengths.as_array().to_owned() };

    let result = extract_data_segment(applicable_data.view(), time_data.view(), &begins, &lengths, time_of_trigger);

    // Convert the Rust Vec<f32> to a NumPy array and return it
    let np_result = Array1::from(result).into_pyarray(py).to_owned();
    Ok(np_result)
}

#[pymodule]
fn qpmfli_downsample_rust(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_extract_data, m)?)?;
    Ok(())
}
