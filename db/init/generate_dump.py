#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IE_project 데이터베이스 SQL 덤프 생성 스크립트
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# DB 모듈 import
from db_query import DatabaseConnection

def generate_sql_dump():
    """Python을 사용해서 SQL 덤프 생성"""
    
    # .env 파일 로드
    load_dotenv()
    
    # DB 연결
    db = DatabaseConnection()
    if not db.connect():
        print(" 데이터베이스 연결 실패")
        return False
    
    try:
        # 덤프 파일 생성
        dump_filename = f"dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        with open(dump_filename, 'w', encoding='utf-8') as f:
            # 헤더 작성
            f.write(f"-- MySQL dump of IE_project database\n")
            f.write(f"-- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"-- Database: ie_project\n")
            f.write(f"--\n\n")
            
            f.write("SET NAMES utf8mb4;\n")
            f.write("SET time_zone = '+00:00';\n")
            f.write("SET foreign_key_checks = 0;\n")
            f.write("SET sql_mode = 'NO_AUTO_VALUE_ON_ZERO';\n\n")
            
            f.write("DROP DATABASE IF EXISTS `ie_project`;\n")
            f.write("CREATE DATABASE `ie_project` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;\n")
            f.write("USE `ie_project`;\n\n")
            
            # 테이블 목록 가져오기
            tables_result = db.execute_query("SHOW TABLES")
            if tables_result is not None:
                tables = tables_result.iloc[:, 0].tolist()
                
                for table in tables:
                    print(f" 덤프 중: {table} 테이블...")
                    
                    # CREATE TABLE 구문 가져오기
                    create_result = db.execute_query(f"SHOW CREATE TABLE `{table}`")
                    if create_result is not None:
                        create_sql = create_result.iloc[0, 1]
                        f.write(f"-- Table structure for table `{table}`\n")
                        f.write(f"DROP TABLE IF EXISTS `{table}`;\n")
                        f.write(f"{create_sql};\n\n")
                    
                    # 데이터 덤프
                    data_result = db.execute_query(f"SELECT * FROM `{table}`")
                    if data_result is not None and len(data_result) > 0:
                        f.write(f"-- Dumping data for table `{table}`\n")
                        
                        # 컬럼명 가져오기
                        columns = list(data_result.columns)
                        columns_str = "`, `".join(columns)
                        
                        f.write(f"LOCK TABLES `{table}` WRITE;\n")
                        f.write(f"/*!40000 ALTER TABLE `{table}` DISABLE KEYS */;\n")
                        
                        # INSERT 구문 생성
                        for _, row in data_result.iterrows():
                            values = []
                            for val in row:
                                if val is None or str(val) == 'nan':
                                    values.append('NULL')
                                elif isinstance(val, str):
                                    # SQL 인젝션 방지를 위한 이스케이프
                                    escaped_val = val.replace("'", "\\'").replace("\\", "\\\\")
                                    values.append(f"'{escaped_val}'")
                                else:
                                    values.append(str(val))
                            
                            values_str = ", ".join(values)
                            f.write(f"INSERT INTO `{table}` (`{columns_str}`) VALUES ({values_str});\n")
                        
                        f.write(f"/*!40000 ALTER TABLE `{table}` ENABLE KEYS */;\n")
                        f.write(f"UNLOCK TABLES;\n\n")
                    else:
                        f.write(f"-- No data to dump for table `{table}`\n\n")
            
            f.write("SET foreign_key_checks = 1;\n")
            f.write("-- Dump completed\n")
        
        print(f" SQL 덤프가 성공적으로 생성되었습니다: {dump_filename}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(dump_filename)
        print(f" 파일 크기: {file_size:,} bytes")
        
        return True
        
    except Exception as e:
        print(f" 덤프 생성 중 오류 발생: {e}")
        return False
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    print("="*60)
    print("IE_project 데이터베이스 SQL 덤프 생성")
    print("="*60)
    
    success = generate_sql_dump()
    if success:
        print("\n 덤프 완료!")
    else:
        print("\n 덤프 실패!")