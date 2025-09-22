"""
ECOS 데이터 전처리 FastAPI 실행 및 테스트 스크립트
"""

import requests
import json
import os

# FastAPI 서버 URL
BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """API 엔드포인트들을 테스트"""
    
    print("=" * 60)
    print("ECOS 데이터 전처리 API 테스트")
    print("=" * 60)
    
    # 1. Health Check
    print("\n1. Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ Health Check 성공")
            print(f"  응답: {response.json()}")
        else:
            print(f"✗ Health Check 실패: {response.status_code}")
    except Exception as e:
        print(f"✗ Health Check 오류: {e}")
    
    # 2. Root endpoint
    print("\n2. Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✓ Root endpoint 성공")
            print(f"  응답: {response.json()}")
        else:
            print(f"✗ Root endpoint 실패: {response.status_code}")
    except Exception as e:
        print(f"✗ Root endpoint 오류: {e}")
    
    # 3. 전처리 실행
    print("\n3. ECOS 데이터 전처리 실행")
    try:
        response = requests.post(f"{BASE_URL}/preprocess")
        if response.status_code == 200:
            result = response.json()
            print("✓ 전처리 성공")
            print(f"  처리된 행 수: {result.get('processed_rows')}")
            print(f"  피쳐 개수: {result.get('feature_count')}")
            print(f"  날짜 범위: {result.get('date_range')}")
        else:
            print(f"✗ 전처리 실패: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"  오류 내용: {error_detail}")
            except:
                print(f"  오류 내용: {response.text}")
    except Exception as e:
        print(f"✗ 전처리 오류: {e}")
    
    # 4. 피쳐 정보 조회
    print("\n4. 피쳐 정보 조회")
    try:
        response = requests.get(f"{BASE_URL}/features/info")
        if response.status_code == 200:
            result = response.json()
            print("✓ 피쳐 정보 조회 성공")
            print(f"  총 피쳐 개수: {result.get('total_features')}")
            print(f"  총 데이터 행 수: {result.get('total_rows')}")
            print(f"  날짜 범위: {result.get('date_range')}")
        else:
            print(f"✗ 피쳐 정보 조회 실패: {response.status_code}")
    except Exception as e:
        print(f"✗ 피쳐 정보 조회 오류: {e}")
    
    # 5. 데이터 미리보기
    print("\n5. 데이터 미리보기")
    try:
        response = requests.get(f"{BASE_URL}/data/preview?limit=5")
        if response.status_code == 200:
            result = response.json()
            print("✓ 데이터 미리보기 성공")
            print(f"  총 행 수: {result.get('total_rows')}")
            print(f"  미리보기 행 수: {result.get('preview_rows')}")
            if result.get('data'):
                print("  샘플 데이터:")
                for i, row in enumerate(result['data'][:2], 1):
                    print(f"    {i}. date: {row.get('date', 'N/A')}")
        else:
            print(f"✗ 데이터 미리보기 실패: {response.status_code}")
    except Exception as e:
        print(f"✗ 데이터 미리보기 오류: {e}")
    
    print("\n" + "=" * 60)
    print("API 테스트 완료")
    print("=" * 60)

def main():
    """메인 함수"""
    print("ECOS 데이터 전처리 FastAPI 테스트")
    print("API 서버가 실행 중인지 확인하세요: python preprocessing.py")
    print("또는: uvicorn preprocessing:app --reload")
    
    # Enter 입력 대기를 제거하고 바로 테스트 실행
    print("\nAPI 테스트를 시작합니다...")
    
    test_api_endpoints()

if __name__ == "__main__":
    main()