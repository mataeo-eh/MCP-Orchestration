### **Available Endpoints**


1. **Activities**
   * **GET** `/api/data/v1/activities`
     * Retrieve a list of activities.
     * Response includes activity details like ID, title, start, and end dates.
2. **Cases**
   * **GET** `/api/data/v1/cases`
     * Retrieve a list of cases.
     * Response includes case details such as patient name, presenting complaint, case ID, and case number.
3. **Events of an Activity**
   * **GET** `/api/data/v1/events/activity/{ActivityID}`
     * Retrieve events related to a specific activity.
     * Requires ActivityID as input.
4. **Nbome Events**
   * **GET** `/api/data/v1/events/nbome`
     * Retrieve events marked as ready for NBOME integration.
5. **Student Assessments (NBOME)**
   * **GET** `/api/data/v1/student-assessments/nbome/event/{EventID}`
     * Retrieve student assessment results for a specific event.
       The results adhere to the NBOME format.

       
:::info
        *IMPORTANT:* If **two SPs** are assigned to **one station** of a scheduled event, every [Case item](/s/ent-hs/doc/items-Jg8T3db0qZ) of the [section](/s/ent-hs/doc/sections-5ToJRGE5sd) will appear duplicated in the Data API (*one with the name of the first SP, the other with the name of the second SP, respectively*).

       :::
6. **Student Video Recordings for NBOME**
   * **GET** `/api/data/v1/student-video-recordings/nbome/event/{EventID}`
     * Retrieve student video recordings for a specific NBOME event.
7. **Students**
   * **GET** `/api/data/v1/students`
     * Retrieve a list of students.
     * Response includes student details like ID, full name, and NBOME ID.
8. **Student Video Recordings for Event and Case**
   * **GET** `/api/data/v1/student-video-recordings/event/{EventID}/case/{CaseID}/student/{StudentID}`
     * Retrieve video recordings for a specific student, event, and case.
9. **Video Files for Video Recording**
   * **GET** `/api/data/v1/video-files/video-recording/{VideoRecordingID}`
     * Retrieve a list of video files associated with a specific video recording.

       
:::success
       When requesting **video file content**, you will be provided with **links to** the video recording files in MP4 format**.**

       :::

### **Response Format**

Each response will include the following:

* **success**: A boolean indicating if the request was successful.
* **total**: The total number of records returned.
* **offset**: The offset value, useful for pagination.
* **limit**: The number of records returned in the current request.
* **data**: An array containing the data objects related to the request.