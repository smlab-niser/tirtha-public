# Tirtha Views Test Suite Documentation

## Overview

This documentation provides a comprehensive guide to the test suite for Tirtha's `views.py` module. The test suite ensures robust functionality for all web views, authentication flows, file uploads, search capabilities, and ARK resolution in the Tirtha cultural heritage digitization platform.

## Running the Tests

### Command to Execute Views Tests

```bash
DJANGO_SETTINGS_MODULE=tirtha_bk.settings .venv/bin/python -m pytest tirtha_bk/tirtha/tests/test_views_optimized.py -v --tb=short --override-ini django_find_project=true
```

### Why This Specific Command is Required

The `pytest.ini` configuration was set up for AliceVision tests, which sets `django_find_project = false`. However, Django views tests require `django_find_project = true` to enable Django functionality. The `--override-ini django_find_project=true` parameter temporarily overrides this setting for the Django test run without affecting the AliceVision tests.

**Note**: The `DJANGO_SETTINGS_MODULE` environment variable is technically optional since `pytest.ini` already contains `DJANGO_SETTINGS_MODULE = tirtha_bk.settings`. However, using the environment variable makes the command more explicit and ensures proper Django configuration regardless of the pytest.ini state.

## Test Files Structure

### Main Test Files
- **`test_views_optimized.py`** - Primary comprehensive test suite (1,654 lines)
- **`test_views_comprehensive.py`** - Alternative comprehensive test suite 
- **`test_views.py`** - Basic test file

### Test Organization
- **49 Django view tests** total in optimized suite
- **3 database integration tests** (previously skipped, now active)
- **Performance and timing tests**
- **Edge case and error handling tests**

## Test Classes and Functions

### 1. TestAuthenticateUser
**Purpose**: Tests the `_authenticate_user` helper function that handles user authentication flow.

#### Key Test Functions:
- **`test_authenticate_user_google_disabled()`**
  - Tests authentication when Google OAuth is disabled
  - Verifies admin fallback authentication
  - Ensures proper admin contributor retrieval

- **`test_authenticate_user_no_user_info()`**
  - Tests behavior with missing user information
  - Verifies proper error handling for null user data

- **`test_authenticate_user_no_email()`**
  - Tests authentication with incomplete user data (missing email)
  - Ensures validation of required user fields

- **`test_authenticate_user_new_contributor()`**
  - Tests creation of new contributors during first-time authentication
  - Verifies proper contributor initialization with inactive status

- **`test_authenticate_user_existing_contributor_name_update()`**
  - Tests updating existing contributor names when changed in OAuth provider
  - Ensures data synchronization between OAuth and local database

- **`test_authenticate_user_banned()`**
  - Tests authentication flow for banned users
  - Verifies proper rejection with appropriate messaging

- **`test_authenticate_user_inactive()`**
  - Tests authentication for inactive (not yet approved) users
  - Ensures proper handling of pending approval status

#### Fixtures:
- `mock_contributor` - Standard active contributor
- `inactive_contributor` - Inactive/pending contributor
- `banned_contributor` - Banned contributor

### 2. TestGetMeshContext
**Purpose**: Tests the `_get_mesh_context` helper function that prepares mesh data for template rendering.

#### Key Test Functions:
- **`test_get_mesh_context_without_run()`**
  - Tests context creation for meshes without associated reconstruction runs
  - Verifies proper mesh metadata extraction (orientation, contribution counts, etc.)
  - Ensures correct static file path generation

- **`test_get_mesh_context_with_run()`**
  - Tests context creation when a specific reconstruction run is provided
  - Verifies run-specific data overrides mesh defaults
  - Tests ARK URL generation for runs
  - Validates runs_arks list creation

#### Key Context Fields Tested:
- `mesh` - The mesh object
- `mesh_contribution_count` - Number of contributions to the mesh
- `mesh_images_count` - Total images associated with the mesh
- `orientation` - 3D rotation values for viewer positioning
- `src` - Path to 3D model file (.glb)
- `run` - Associated reconstruction run (if any)
- `run_contributor_count` - Contributors to specific run
- `run_images_count` - Images used in specific run
- `run_ark_url` - Full URL for ARK resolution

### 3. TestIndexView
**Purpose**: Tests the main index/homepage view that displays 3D models and handles various routing scenarios.

#### Key Test Functions:
- **`test_index_basic()`**
  - Tests basic homepage loading without specific mesh or run parameters
  - Verifies proper template rendering with default mesh
  - Ensures mesh list population

- **`test_index_with_runid()`**
  - Tests loading specific reconstruction runs via URL parameter
  - Verifies run-to-mesh relationship handling
  - Tests proper context switching to run-specific data

- **`test_index_run_not_found()`**
  - Tests error handling for non-existent run IDs
  - Verifies proper 404 handling

- **`test_index_with_vid()`**
  - Tests loading specific meshes via verbose ID parameter
  - Verifies mesh lookup and latest run association

- **`test_index_with_authenticated_user()`**
  - Tests homepage behavior for signed-in users
  - Verifies proper user session handling
  - Tests profile image and authentication status display

### 4. TestSigninView
**Purpose**: Tests OAuth2 authentication initiation flow.

#### Key Test Functions:
- **`test_signin()`**
  - Tests OAuth2 flow initiation
  - Verifies state parameter generation for security
  - Tests redirect URL construction
  - Validates session state storage

### 5. TestVerifyTokenView
**Purpose**: Tests OAuth2 callback handling and token verification.

#### Key Test Functions:
- **`test_verify_token_state_mismatch()`**
  - Tests security validation of OAuth state parameter
  - Verifies proper rejection of tampered requests

- **`test_verify_token_no_state()`**
  - Tests handling of malformed OAuth callbacks
  - Ensures security through proper validation

- **`test_verify_token_success()`**
  - Tests successful OAuth token verification
  - Verifies user session creation
  - Tests user information extraction from JWT tokens

- **`test_verify_token_invalid_token()`**
  - Tests handling of invalid or expired tokens
  - Verifies proper error handling and security

### 6. TestPreUploadCheckView
**Purpose**: Tests pre-upload validation that checks user permissions and mesh availability.

#### Key Test Functions:
- **`test_pre_upload_check_no_mesh_id()`**
  - Tests validation when mesh ID is missing from request
  - Verifies proper error messaging

- **`test_pre_upload_check_no_user_session()`**
  - Tests behavior for unauthenticated users
  - Verifies authentication requirement enforcement

- **`test_pre_upload_check_auth_failed()`**
  - Tests handling of authentication failures
  - Verifies proper rejection with blur UI state

- **`test_pre_upload_check_mesh_not_found()`**
  - Tests validation for non-existent meshes
  - Verifies database lookup error handling

- **`test_pre_upload_check_mesh_completed()`**
  - Tests validation for meshes that no longer accept contributions
  - Verifies proper business rule enforcement

- **`test_pre_upload_check_success()`**
  - Tests successful validation flow
  - Verifies permission to upload is granted

#### Response Format:
```json
{
  "allowupload": boolean,
  "blur": boolean,
  "output": "status message"
}
```

### 7. TestUploadView
**Purpose**: Tests the main file upload functionality for contributing images to mesh reconstructions.

#### Key Test Functions:
- **`test_upload_auth_failed()`**
  - Tests upload rejection for unauthenticated users
  - Verifies security enforcement

- **`test_upload_banned_user()`**
  - Tests upload rejection for banned users
  - Verifies proper access control

- **`test_upload_no_mesh_id()`**
  - Tests validation of required mesh ID parameter
  - Verifies proper error handling

- **`test_upload_mesh_not_found()`**
  - Tests handling of uploads to non-existent meshes
  - Verifies database validation

- **`test_upload_no_images()`**
  - Tests validation when no image files are provided
  - Verifies file upload requirement enforcement

- **`test_upload_success()`**
  - Tests complete successful upload flow
  - Verifies contribution creation in database
  - Tests bulk image creation
  - Validates background task scheduling for image processing

#### Upload Flow:
1. User authentication validation
2. Mesh existence verification
3. User permission checks (not banned, active)
4. File validation
5. Contribution record creation
6. Bulk image record creation
7. Background task scheduling for image processing

### 8. TestSearchView
**Purpose**: Tests the search functionality for finding meshes by various criteria.

#### Key Test Functions:
- **`test_search_no_query()`**
  - Tests behavior when no search query is provided
  - Verifies proper empty result handling

- **`test_search_empty_query()`**
  - Tests handling of whitespace-only search queries
  - Verifies input sanitization

- **`test_search_no_results()`**
  - Tests response when no meshes match the search criteria
  - Verifies proper "not found" messaging

- **`test_search_with_results()`**
  - Tests successful search with matching results
  - Verifies proper JSON formatting of mesh data
  - Tests completion status color coding

#### Search Response Format:
```json
{
  "status": "Mesh found!" | "Mesh not found!" | "No query provided",
  "meshes_json": {
    "Mesh Name": {
      "verbose_id": "mesh_id",
      "url": "/media/thumbnail.jpg",
      "completed_col": "forestgreen" | "firebrick"
    }
  }
}
```

#### Search Criteria:
- Mesh name (partial matching)
- Country
- State
- District

### 9. TestResolveARKView
**Purpose**: Tests ARK (Archival Resource Key) resolution for persistent identifiers.

#### Key Test Functions:
- **`test_resolve_ark_success()`**
  - Tests successful ARK resolution to mesh/run
  - Verifies proper redirect to specific mesh view
  - Tests ARK parsing and database lookup

- **`test_resolve_ark_parse_error()`**
  - Tests handling of malformed ARK identifiers
  - Verifies fallback to external ARK resolver

- **`test_resolve_ark_not_found()`**
  - Tests handling of valid but non-existent ARKs
  - Verifies proper fallback mechanism

#### ARK Format:
- Standard: `ark:/naan/assigned_name`
- Example: `ark:/12345/mesh_run_identifier`

### 10. TestStaticViews
**Purpose**: Tests static page views (competition, howto).

#### Key Test Functions:
- **`test_competition_view()`**
  - Tests competition information page rendering
  - Verifies proper template selection

- **`test_howto_view()`**
  - Tests user guide page rendering
  - Verifies proper template selection

### 11. TestDatabaseIntegration (Django DB Tests)
**Purpose**: Tests that require actual database operations and model interactions.

#### Key Test Functions:
- **`test_authenticate_user_database_integration()`**
  - Tests authentication with real database operations
  - Verifies contributor creation and updates
  - Tests active/inactive status handling

- **`test_index_view_database_integration()`**
  - Tests homepage with actual database queries
  - Verifies mesh loading from database

- **`test_upload_view_database_integration()`**
  - Tests upload functionality with real database
  - Verifies contribution and image record creation

### 12. Performance and Edge Case Tests

#### TestViewsPerformance:
- **`test_search_performance_large_dataset()`**
  - Tests search functionality with 1000+ mesh records
  - Verifies response time under load (< 2 seconds)

- **`test_upload_performance_many_images()`**
  - Tests upload handling with 50+ images
  - Verifies bulk operations efficiency (< 3 seconds)

#### TestViewsTiming:
- **`test_index_response_time()`**
  - Tests homepage load time (< 1 second)
  - Verifies acceptable user experience

#### TestEdgeCases:
- **`test_large_search_query()`**
  - Tests handling of 10,000+ character search queries
  - Verifies input length handling

- **`test_malformed_session_data()`**
  - Tests resilience against corrupted session data
  - Verifies graceful error handling

- **`test_upload_task_failure()`**
  - Tests handling when background image processing fails
  - Verifies proper cleanup and error messaging

## Test Infrastructure

### Mocking Strategy
The test suite uses comprehensive mocking to avoid external dependencies:

#### Django Framework Mocks:
- **RequestFactory** - Creates HTTP requests for testing
- **SessionStore** - Mocks user sessions
- **JsonResponse** - Mocks JSON API responses

#### Database Mocks:
- **Model.objects.get/filter** - Mocks database queries
- **Mock contributors, meshes, runs** - Test data objects

#### File System Mocks:
- **PIL Image operations** - Prevents actual image processing
- **File upload handling** - Mocks multipart form data
- **Static file paths** - Prevents file system access

#### External Service Mocks:
- **Google OAuth2** - Mocks authentication flow
- **Background tasks** - Mocks Celery task scheduling

### Conditional Patching System
The test suite implements a conditional patching system that only applies Django-specific mocks when running Django database tests:

```python
RUNNING_DJANGO_DB_TESTS = 'DJANGO_SETTINGS_MODULE' in os.environ

if RUNNING_DJANGO_DB_TESTS:
    # Apply patches for file operations, image processing, etc.
    # Only when Django settings are active
```

This ensures:
- ✅ Django tests get necessary mocks for file operations
- ✅ AliceVision tests run without interference
- ✅ Perfect test isolation between different test suites

### Custom Mock Classes

#### MockPILImage:
Comprehensive mock for PIL/Pillow image operations:
- Image dimensions (width, height, size)
- Image operations (save, load, copy, rotate, resize, close)
- Prevents actual image file processing during tests

## Test Results and Coverage

### Current Status:
- ✅ **49/49 Django view tests passing** (100% success rate)
- ✅ **3/3 database integration tests active** (previously skipped)
- ✅ **All authentication flows tested**
- ✅ **Complete upload pipeline coverage**
- ✅ **ARK resolution functionality verified**
- ✅ **Performance benchmarks met**

### Test Categories:
- **Unit Tests**: Individual function testing (80% of suite)
- **Integration Tests**: Database interaction testing (6% of suite)
- **Performance Tests**: Load and timing validation (8% of suite)
- **Edge Case Tests**: Error handling and resilience (6% of suite)

### Error Scenarios Covered:
- Authentication failures
- Invalid input validation
- Database lookup failures
- File upload errors
- Session corruption
- External service failures
- Performance degradation
- Security violations

## Key Testing Patterns

### 1. Request/Response Testing:
```python
request = request_factory.get("/path/")
response = view_function(request)
assert isinstance(response, JsonResponse)
data = json.loads(response.content)
assert data["status"] == "expected_value"
```

### 2. Authentication Flow Testing:
```python
with patch("tirtha.views._authenticate_user") as mock_auth:
    mock_auth.return_value = ("Success", mock_contributor)
    response = protected_view(request)
    # Verify authentication was called and response is correct
```

### 3. Database Integration Testing:
```python
@pytest.mark.django_db
def test_with_real_database():
    # Test interacts with actual Django database
    contributor = Contributor.objects.create(...)
    # Test database operations
```

### 4. File Upload Testing:
```python
from django.core.files.uploadedfile import SimpleUploadedFile
image_file = SimpleUploadedFile("test.jpg", b"content", "image/jpeg")
request.FILES.getlist = Mock(return_value=[image_file])
```

## Security Testing

### Authentication Security:
- OAuth2 state parameter validation
- Session hijacking prevention
- Token verification and expiry
- User permission enforcement

### Input Validation:
- SQL injection prevention through ORM usage
- File upload size and type restrictions
- Query parameter sanitization
- Session data integrity

### Access Control:
- User authentication requirements
- Banned user restrictions
- Inactive user handling
- Admin privilege separation

## Maintenance and Future Development

### Adding New Tests:
1. Follow existing class structure (`TestNewFeature`)
2. Use appropriate fixtures for setup
3. Mock external dependencies
4. Include both success and failure scenarios
5. Add performance tests for critical paths

### Test Data Management:
- Use fixtures for reusable test data
- Mock database operations to avoid state pollution
- Clean up resources in teardown methods
- Use temporary directories for file operations

### Performance Considerations:
- Keep test execution time under 30 seconds total
- Use parallel test execution where possible
- Mock expensive operations (file I/O, external APIs)
- Monitor test suite performance over time

This comprehensive test suite ensures the Tirtha platform's web interface is robust, secure, and performant for cultural heritage digitization workflows.
