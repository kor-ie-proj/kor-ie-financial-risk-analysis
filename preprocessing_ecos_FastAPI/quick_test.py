#!/usr/bin/env python3
"""
빠른 API 테스트 스크립트
"""
import requests
import json
import time

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Health check
    print("=== Health Check ===")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Health check 실패: {e}")
        return
    
    # 2. Root endpoint
    print("=== Root Endpoint ===")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Root endpoint 실패: {e}")
    
    # 3. Load ECOS data
    print("=== Load ECOS Data ===")
    try:
        response = requests.post(f"{base_url}/load-ecos-data")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ ECOS 데이터 로드 성공!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ ECOS 데이터 로드 실패: {response.text}")
        print()
    except Exception as e:
        print(f"ECOS 데이터 로드 실패: {e}")
    
    # 4. Preprocessing
    print("=== Preprocessing ===")
    try:
        response = requests.post(f"{base_url}/preprocessing")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ 전처리 성공!")
            result = response.json()
            print(f"처리된 행 수: {result.get('processed_rows', 'N/A')}")
            print(f"피처 개수: {result.get('feature_count', 'N/A')}")
        else:
            print(f"❌ 전처리 실패: {response.text}")
        print()
    except Exception as e:
        print(f"전처리 실패: {e}")

if __name__ == "__main__":
    test_api()