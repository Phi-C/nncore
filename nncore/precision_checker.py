#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from logging import exception
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Type

import torch
from torch.nn.functional import cosine_similarity

from rich.table import Table
from rich.console import Console


class PrecisionChecker:
    """A class for comparing tensor precision between standard and observed data."""
    
    def __init__(
        self, 
        standard_dir: str, 
        observed_dir: str, 
        name_list: Optional[List[str]] = None, 
        eps: float = 1e-8
    ):
        """
        Initialize the PrecisionChecker.
        
        Args:
            standard_dir: Directory containing standard tensor files
            observed_dir: Directory containing observed tensor files
            name_list: Optional list of specific files to compare
            eps: Small value to avoid division by zero
        """
        self.standard_dir = standard_dir
        self.observed_dir = observed_dir
        self.eps = eps
        self.name_list = self._get_name_list() if name_list is None else name_list
        self.table = self._prepare_table()
        self.console = Console()
        
    def _get_name_list(self) -> List[str]:
        """Get sorted list of .pth files in observed directory."""
        return sorted(Path(self.observed_dir).glob("*.pth"), key=lambda x: x.name)
    
    def _prepare_table(self) -> Table:
        """Prepare the rich Table for displaying results."""
        table = Table(title="统计指标")
        # Add columns with styles
        columns = [
            ("Name", "bold red"),
            ("MRE", "cyan"),
            ("SRN", "magenta"),
            ("MEAN(Stand vs Obser)", "green"),
            ("VAR(Stand vs Obser)", "yellow"),
            ("COS", "cyan"),
            ("CCC", "blue"),
            ("sMAPE", "red"),
            ("Person", "white"),
            ("Ulp_Max_Diff", "green"),
        ]
        for name, style in columns:
            table.add_column(name, style=style)
        return table
    
    @staticmethod
    def _check_mre(observed: torch.Tensor, standard: torch.Tensor, eps: float) -> float:
        """Calculate Mean Relative Error."""
        relative_err = torch.abs(observed - standard) / (torch.abs(standard) + eps)
        return torch.mean(relative_err).item()
    
    @staticmethod
    def _check_snr(observed: torch.Tensor, standard: torch.Tensor, eps: float) -> float:
        """Calculate Signal-to-Noise Ratio."""
        noise = observed.float() - standard.float()
        signal_power = torch.mean(standard.float() ** 2)
        noise_power = torch.mean(noise ** 2)
        return (10 * torch.log10(signal_power / (noise_power + eps))).item()
    
    @staticmethod
    def _check_person(observed: torch.Tensor, standard: torch.Tensor) -> float:
        """Calculate Pearson correlation coefficient."""
        observed = observed.float()
        standard = standard.float()
        
        x_centered = observed - torch.mean(observed)
        y_centered = standard - torch.mean(standard)
        
        covariance = torch.sum(x_centered * y_centered)
        std_x = torch.sqrt(torch.sum(x_centered ** 2))
        std_y = torch.sqrt(torch.sum(y_centered ** 2))
        
        if std_x == 0 or std_y == 0:
            return 0.0
            
        return (covariance / (std_x * std_y)).item()
    
    @staticmethod
    def _check_ccc(observed: torch.Tensor, standard: torch.Tensor) -> float:
        """Calculate Concordance Correlation Coefficient."""
        observed = observed.float()
        standard = standard.float()
        
        x_mean, y_mean = torch.mean(observed), torch.mean(standard)
        covariance = torch.mean((observed - x_mean) * (standard - y_mean))
        x_var = torch.var(observed, unbiased=False)
        y_var = torch.var(standard, unbiased=False)
        
        numerator = 2 * covariance
        denominator = x_var + y_var + (x_mean - y_mean) ** 2
        
        return (numerator / denominator).item() if denominator != 0 else 1.0
    
    @staticmethod
    def _check_smape(observed: torch.Tensor, standard: torch.Tensor, eps: float = 1e-8) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        observed, standard = observed.float(), standard.float()
        denominator = torch.abs(observed) + torch.abs(standard)
        denominator = torch.where(denominator == 0, torch.tensor(eps), denominator)
        return (100 * torch.mean(2 * torch.abs(standard - observed) / denominator)).item()
    
    @staticmethod
    def _check_cos_sim(observed: torch.Tensor, standard: torch.Tensor) -> Tuple[float, int]:
        """Calculate minimum cosine similarity between tensor rows."""
        observed, standard = observed.float(), standard.float()

        standard_mat = standard.reshape(-1, standard.shape[-1])
        observed_mat = observed.reshape(-1, observed.shape[-1])
        similarities = cosine_similarity(standard_mat, observed_mat, dim=1)
        min_sim, min_idx = torch.min(similarities), torch.argmin(similarities)
        return min_sim.item(), min_idx.item()

    # TODO: 检查实现
    @staticmethod
    def _check_ulperr(observed: torch.Tensor, standard: torch.Tensor) -> Tuple[float, float]:
        """Calculate ULP error difference between two tensors."""    
        if observed.dtype != standard.dtype:
            raise TypeError("Input types must match.")
    
        dtype = observed.dtype
        if dtype in {torch.half, torch.bfloat16, torch.float32, torch.double}:
            machine_eps = torch.finfo(dtype).eps
        else:
            raise TypeError(f"Unsupported data type: {dtype}")
        
        # EXP_BIAS = {
        #     torch.half: 15,
        #     torch.bfloat16: 127,
        #     torch.float32: 127,
        #     torch.double: 1023,
        # }
    
        abs_error = torch.abs(observed - standard)
        
        # 处理参考值
        ref = standard.clone()
        
        zero_mask = (ref == 0)
        if zero_mask.any():
            smallest_positive = torch.nextafter(ref[zero_mask], torch.tensor(float('inf'), device=ref.device))
            ref[zero_mask] = smallest_positive
            
        inf_mask = torch.isinf(ref)
        if inf_mask.any():
            ref[inf_mask] = torch.finfo(dtype).max
        
        # 计算
        exponents = torch.floor(torch.log2(torch.abs(ref)))
        ulp_standard = torch.pow(2.0, exponents) * machine_eps
        ulp_error = torch.where(ulp_standard > 0, abs_error / ulp_standard, torch.zeros_like(abs_error))
        
        # 处理NaN和InF的情况
        ulp_error = torch.where(torch.isnan(ulp_error), torch.tensor(float('inf'), device=ulp_error.device), ulp_error)
        
        return torch.max(ulp_error).item(), torch.mean(ulp_error).item()
    
    @staticmethod
    def _save_plot(data: List[float], xlabel: str, ylabel: str, title: str, filename: str):
        """Helper method to save plots."""
        plt.figure()
        plt.scatter(range(len(data)), data, s=1, color="blue", label="Data Points")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
    
    def _show_scatter_plot(self, observed: List[float], standard: List[float], tensor_name: str):
        """Generate scatter plot comparing standard vs observed."""
        self._save_plot(
            data=standard,
            xlabel="GPU Values",
            ylabel="XPU Values",
            title=f"Scatter Plot for {tensor_name}",
            filename=f"{tensor_name}_scatter.png"
        )
    
    def _show_abserr_plot(self, standard: List[float], observed: List[float], tensor_name: str):
        """Generate absolute error plot."""
        res = [s - o for s, o in zip(standard, observed)]
        self._save_plot(
            data=res,
            xlabel="Index Values",
            ylabel="Absolute Error Values",
            title=f"Absolute error for {tensor_name}",
            filename=f"{tensor_name}_abserr.png"
        )
    
    def _show_relerr_plot(self, standard: List[float], observed: List[float], tensor_name: str):
        """Generate relative error plot."""
        res = [abs(s - o) / (abs(s) + self.eps) for s, o in zip(standard, observed)]
        self._save_plot(
            data=res,
            xlabel="Index Values",
            ylabel="Relative Error Values",
            title=f"Relative error for {tensor_name}",
            filename=f"{tensor_name}_relerr.png"
        )
    
    def _validate_tensor(self, tensor: torch.Tensor, name: str):
        """Check for NaN or Inf values in tensor."""
        if torch.isnan(tensor).any():
            raise ValueError(f"Observed data {name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"Observed data {name} contains Inf values")
    
    def check(self, verbose: bool = True, show_plots: bool = False):
        """Compare all tensors between standard and observed directories."""
        for name in self.name_list:
            try:
                standard_path = Path(self.standard_dir) / name.name
                observed_path = Path(self.observed_dir) / name.name
                
                standard_tensor = torch.load(standard_path)
                observed_tensor = torch.load(observed_path)
                
                self._validate_tensor(observed_tensor, name.name)
                
                # Calculate metrics
                metrics = {
                    "mre": self._check_mre(observed_tensor, standard_tensor, self.eps),
                    "snr": self._check_snr(observed_tensor, standard_tensor, self.eps),
                    "ccc": self._check_ccc(observed_tensor, standard_tensor),
                    "person": self._check_person(observed_tensor, standard_tensor),
                    "smape": self._check_smape(observed_tensor, standard_tensor, self.eps),
                    "cos_sim": self._check_cos_sim(observed_tensor, standard_tensor)[0],
                    "standard_mean": torch.mean(standard_tensor).item(),
                    "observed_mean": torch.mean(observed_tensor).item(),
                    "standard_var": torch.var(standard_tensor).item(),
                    "observed_var": torch.var(observed_tensor).item(),
                    "ulp_diff": self._check_ulperr(observed_tensor, standard_tensor)[0],
                }
                
                # Add row to table
                self.table.add_row(
                    name.name,
                    f"{metrics['mre']:.4f}",
                    f"{metrics['snr']:.2f}",
                    f"{metrics['standard_mean']:.4f}-{metrics['observed_mean']:.4f}",
                    f"{metrics['standard_var']:.4f}-{metrics['observed_var']:.4f}",
                    f"{metrics['cos_sim']:.4f}",
                    f"{metrics['ccc']:.4f}",
                    f"{metrics['person']:.4f}",
                    f"{metrics['smape']:.4f}",
                    f"{metrics['ulp_diff']:.2f}"
                )
                
                if verbose:
                    self.console.print(
                        f"{name.name}: MRE = {metrics['mre']:.4f}, "
                        f"SNR = {metrics['snr']:.2f}, "
                        f"CCC = {metrics['ccc']:.4f}, "
                        f"PERSON = {metrics['person']:.4f}, "
                        f"SMAPE = {metrics['smape']:.4f}, "
                        f"ULP_DIFF = {metrics['ulp_diff']:.2f}"
                    )
                
                if show_plots:
                    standard_data = standard_tensor.flatten().cpu().float().tolist()
                    observed_data = observed_tensor.flatten().cpu().float().tolist()
                    self._show_scatter_plot(standard_data, observed_data, name.name)
                    self._show_abserr_plot(standard_data, observed_data, name.name)
                    self._show_relerr_plot(standard_data, observed_data, name.name)
                    
            except Exception as e:
                self.console.print(f"[red]Error processing {name.name}: {str(e)}[/red]")
                continue
        
        self.console.print(self.table)


def main():
    parser = argparse.ArgumentParser(description="Compare tensor precision between standard and observed data.")
    parser.add_argument("--standard", type=str, default="../data/standard", help="Directory with standard tensors")
    parser.add_argument("--observed", type=str, default="../data/observed", help="Directory with observed tensors")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--plots", action="store_true", help="Generate comparison plots")
    args = parser.parse_args()
    
    checker = PrecisionChecker(args.standard, args.observed)
    checker.check(verbose=args.verbose, show_plots=args.plots)


if __name__ == "__main__":
    main()
