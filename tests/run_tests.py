"""Test runner for MNIST Classifier with detailed reporting."""

import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import json


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    print("🧪 Running MNIST Classifier Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create reports directory
    reports_dir = Path("test_reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Test commands to run
    test_commands = [
        {
            "name": "Unit Tests with Coverage",
            "command": [
                "python", "-m", "pytest", 
                "tests/", 
                "-v", 
                "--cov=mnist_classifier",
                "--cov-report=term-missing",
                "--cov-report=html:test_reports/coverage_html",
                f"--cov-report=json:test_reports/coverage_{timestamp}.json",
                "--tb=short"
            ]
        },
        {
            "name": "API Integration Tests",
            "command": [
                "python", "-m", "pytest",
                "tests/test_api.py",
                "-v",
                "--tb=short"
            ]
        },
        {
            "name": "Model Tests",
            "command": [
                "python", "-m", "pytest",
                "tests/test_models.py",
                "-v", 
                "--tb=short"
            ]
        },
        {
            "name": "Training Tests", 
            "command": [
                "python", "-m", "pytest",
                "tests/test_training.py",
                "-v",
                "--tb=short"
            ]
        }
    ]
    
    results = []
    
    for test_group in test_commands:
        print(f"\n🔄 Running {test_group['name']}...")
        print("-" * 40)
        
        group_start = time.time()
        
        try:
            result = subprocess.run(
                test_group["command"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test group
            )
            
            group_duration = time.time() - group_start
            
            test_result = {
                "name": test_group["name"],
                "command": " ".join(test_group["command"]),
                "duration": group_duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": "PASSED" if result.returncode == 0 else "FAILED"
            }
            
            results.append(test_result)
            
            # Print summary
            if result.returncode == 0:
                print(f"✅ {test_group['name']}: PASSED ({group_duration:.2f}s)")
            else:
                print(f"❌ {test_group['name']}: FAILED ({group_duration:.2f}s)")
                print("STDERR:", result.stderr)
            
            # Print stdout for coverage info and test details
            if result.stdout:
                print(result.stdout)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_group['name']}: TIMEOUT")
            results.append({
                "name": test_group["name"],
                "command": " ".join(test_group["command"]),
                "duration": 300,
                "return_code": -1,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "status": "TIMEOUT"
            })
        except Exception as e:
            print(f"💥 {test_group['name']}: ERROR - {e}")
            results.append({
                "name": test_group["name"],
                "command": " ".join(test_group["command"]),
                "duration": 0,
                "return_code": -2,
                "stdout": "",
                "stderr": str(e),
                "status": "ERROR"
            })
    
    total_duration = time.time() - start_time
    
    # Generate summary
    passed_tests = sum(1 for r in results if r["status"] == "PASSED")
    failed_tests = sum(1 for r in results if r["status"] == "FAILED")
    error_tests = sum(1 for r in results if r["status"] in ["TIMEOUT", "ERROR"])
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Test Groups: {len(results)}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"💥 Errors/Timeouts: {error_tests}")
    print(f"⏱️  Total Duration: {total_duration:.2f}s")
    print(f"📅 Timestamp: {timestamp}")
    
    # Coverage summary
    coverage_file = reports_dir / f"coverage_{timestamp}.json"
    if coverage_file.exists():
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
            print(f"📈 Test Coverage: {total_coverage:.1f}%")
            
            if total_coverage >= 80:
                print("🎯 Coverage target met (≥80%)")
            else:
                print("⚠️  Coverage below target (<80%)")
                
        except Exception as e:
            print(f"⚠️  Could not read coverage data: {e}")
    
    # Save detailed report
    report_data = {
        "timestamp": timestamp,
        "total_duration": total_duration,
        "summary": {
            "total_groups": len(results),
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests
        },
        "results": results
    }
    
    report_file = reports_dir / f"test_results_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📝 Detailed report saved: {report_file}")
    print(f"🌐 Coverage report: test_reports/coverage_html/index.html")
    
    # Exit with error code if any tests failed
    if failed_tests > 0 or error_tests > 0:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1
    else:
        print("\n🎉 All tests passed!")
        return 0


def run_quick_tests():
    """Run a quick subset of tests."""
    print("🚀 Running Quick Test Suite")
    print("=" * 40)
    
    try:
        result = subprocess.run([
            "python", "-m", "pytest",
            "tests/test_models.py::TestMLPClassifier::test_mlp_forward_pass",
            "tests/test_api.py::TestAPIEndpoints::test_health_endpoint",
            "tests/test_metrics.py::TestModelEvaluator::test_basic_metrics_calculation",
            "-v"
        ], capture_output=True, text=True, timeout=60)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✅ Quick tests passed!")
            return 0
        else:
            print("❌ Quick tests failed!")
            return 1
            
    except subprocess.TimeoutExpired:
        print("⏰ Quick tests timed out!")
        return 1
    except Exception as e:
        print(f"💥 Error running quick tests: {e}")
        return 1


def run_lint_checks():
    """Run code quality checks."""
    print("🔍 Running Code Quality Checks")
    print("=" * 40)
    
    checks = [
        {
            "name": "Flake8 (Style)",
            "command": ["python", "-m", "flake8", "mnist_classifier/", "--max-line-length=100"]
        },
        {
            "name": "MyPy (Type Checking)",
            "command": ["python", "-m", "mypy", "mnist_classifier/", "--ignore-missing-imports"]
        }
    ]
    
    all_passed = True
    
    for check in checks:
        print(f"\n🔄 Running {check['name']}...")
        
        try:
            result = subprocess.run(
                check["command"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ {check['name']}: PASSED")
            else:
                print(f"❌ {check['name']}: FAILED")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {check['name']}: TIMEOUT")
            all_passed = False
        except FileNotFoundError:
            print(f"⚠️  {check['name']}: Tool not installed, skipping...")
        except Exception as e:
            print(f"💥 {check['name']}: ERROR - {e}")
            all_passed = False
    
    return 0 if all_passed else 1


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "full"
    
    if mode == "quick":
        return run_quick_tests()
    elif mode == "lint":
        return run_lint_checks()
    elif mode == "full":
        # Run full test suite
        test_result = run_tests_with_coverage()
        
        # Also run lint checks if tests pass
        if test_result == 0:
            print("\n" + "=" * 60)
            lint_result = run_lint_checks()
            return max(test_result, lint_result)
        else:
            return test_result
    else:
        print("Usage: python run_tests.py [full|quick|lint]")
        print("  full  - Run all tests with coverage (default)")
        print("  quick - Run quick subset of tests")
        print("  lint  - Run code quality checks only")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)